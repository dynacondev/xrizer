//! Writer side of the `/xplane-hands-v1` hand-joint feed (xrizer -> hands/ shim).
//!
//! Byte-exact mirror of `hands/src/shm.h` (magic `XPH1`, version 1, 1488 bytes).
//! FROZEN contract: name, layout, and magic must not change; a future layout
//! bumps the segment name instead.
//!
//! Writer discipline (mirrors `reprojex/src/publish.c`): unlink-then-create on
//! init so a fresh run never inherits a dead segment, then per publish bump
//! seq (odd = inside), write all fields, bump seq (even = complete). Readers
//! retry while seq is odd or changed across the read. Single writer; the
//! process-local [`SLOT`] mutex only serializes our own writer threads —
//! readers never take it, so the frame can never block on a reader.
//!
//! Coordinate frame: joints are located relative to xrizer's current tracking
//! space (`SessionData::tracking_space()` — STAGE-adjusted when Standing,
//! LOCAL-adjusted when Seated), in meters, OpenXR right-handed Y-up, in
//! OpenXR joint order (`XrHandJointEXT` 0 = PALM .. 25 = LITTLE_TIP). This is
//! the same frame the WaitGetPoses controller/HMD poses live in. Published
//! as-is (no OpenVR axis swap — that conversion is OpenVR-skeletal specific);
//! hands/ consumes the values directly as its app frame.
//!
//! Failure policy: fail silent. When the segment cannot be mapped (no
//! `/dev/shm`, read-only FS, …) publishing is skipped and a warning is logged
//! once per process; mapping is retried at most once per second. Kill switch:
//! `XRIZER_HANDS_SHM=0/false/no/off` disables publishing (default ON).

use openxr as xr;
use std::ffi::CString;
use std::ptr;
use std::sync::{
    LazyLock, Mutex,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};

use log::warn;

pub const HX_SHM_NAME: &str = "/xplane-hands-v1";
pub const HX_MAGIC: u32 = 0x58504831; // 'XPH1'
pub const HX_VERSION: u32 = 1;
pub const HX_HANDS: usize = 2;
pub const HX_JOINTS: usize = 26;
pub const HX_SIZE: usize = 1488;
pub const HX_LEFT_VALID: u32 = 1 << 0;
pub const HX_RIGHT_VALID: u32 = 1 << 1;

/// Retry a failed mapping at most once per second (mono ns).
const RETRY_NS: u64 = 1_000_000_000;
/// Seqlock read-retry bound for the test reader.
#[cfg(test)]
const READ_TRIES: u32 = 64;

/// One joint: position (meters) + orientation quaternion (x, y, z, w, unit).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct HxJoint {
    pos: [f32; 3],
    quat: [f32; 4],
}

/// Byte-exact mirror of `hx_shm_v1` in `hands/src/shm.h`.
#[repr(C)]
struct HxShmV1 {
    magic: u32,                               // +0
    version: u32,                             // +4
    seq: AtomicU32,                           // +8 seqlock (odd = writer inside)
    flags: u32,                               // +12 validity bits
    frame_mono_ns: u64,                       // +16 CLOCK_MONOTONIC of sample
    generation: u64,                          // +24 writer gen (0 = none yet)
    joints: [[HxJoint; HX_JOINTS]; HX_HANDS], // +32
}

const _: () = assert!(size_of::<HxJoint>() == 28);
const _: () = assert!(size_of::<HxShmV1>() == HX_SIZE);
const _: () = assert!(size_of::<HxShmV1>() == 1488);
const _: () = assert!(std::mem::offset_of!(HxShmV1, magic) == 0);
const _: () = assert!(std::mem::offset_of!(HxShmV1, version) == 4);
const _: () = assert!(std::mem::offset_of!(HxShmV1, seq) == 8);
const _: () = assert!(std::mem::offset_of!(HxShmV1, flags) == 12);
const _: () = assert!(std::mem::offset_of!(HxShmV1, frame_mono_ns) == 16);
const _: () = assert!(std::mem::offset_of!(HxShmV1, generation) == 24);
const _: () = assert!(std::mem::offset_of!(HxShmV1, joints) == 32);

/// Kill switch, default ON. Explicit `0/false/no/off` disables; anything
/// unparseable warns once and stays ON (fail-open toward publishing).
pub fn hands_shm_enabled() -> bool {
    static WARNED: AtomicBool = AtomicBool::new(false);
    match std::env::var("XRIZER_HANDS_SHM") {
        Err(_) => true,
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            match t.as_str() {
                "0" | "false" | "no" | "off" => false,
                "1" | "true" | "yes" | "on" => true,
                _ => {
                    if !WARNED.swap(true, Ordering::Relaxed) {
                        warn!("XRIZER_HANDS_SHM={v:?} unparseable; defaulting ON");
                    }
                    true
                }
            }
        }
    }
}

/// `CLOCK_MONOTONIC` in ns — the same clock `frame_mono_ns` is stamped with.
/// Returns 0 on `clock_gettime` failure.
pub fn now_mono_ns() -> u64 {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if r != 0 {
        return 0;
    }
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

struct Slot {
    ptr: *mut HxShmV1,
    fd: std::os::fd::RawFd,
    last_fail_ns: u64,
}

// Only touched while holding SLOT; the raw pointer is never shared otherwise.
unsafe impl Send for Slot {}

impl Slot {
    const fn new() -> Self {
        Self {
            ptr: ptr::null_mut(),
            fd: -1,
            last_fail_ns: 0,
        }
    }

    fn ensure_locked(&mut self) -> Option<*mut HxShmV1> {
        if !self.ptr.is_null() {
            return Some(self.ptr);
        }
        let now = now_mono_ns();
        if self.last_fail_ns != 0 && now.wrapping_sub(self.last_fail_ns) < RETRY_NS {
            return None;
        }
        match map_shm() {
            Some((fd, shm)) => {
                self.fd = fd;
                self.ptr = shm;
                Some(shm)
            }
            None => {
                self.last_fail_ns = now;
                None
            }
        }
    }
}

static SLOT: LazyLock<Mutex<Slot>> = LazyLock::new(|| Mutex::new(Slot::new()));
/// Monotonic publish counter; stored as `generation` (starts at 1, 0 = none).
static GENERATION: AtomicU64 = AtomicU64::new(0);
static WARNED_SHM: AtomicBool = AtomicBool::new(false);

fn warn_shm_once(msg: String) {
    if !WARNED_SHM.swap(true, Ordering::Relaxed) {
        warn!("{msg}");
    }
}

/// Unlink-then-create + zero, mirroring `reprojex publish_ensure`. The fd is
/// held for the life of the process; the segment is intentionally not unlinked
/// on exit (no cdylib exit hook) — the next run recreates it fresh, and
/// readers must treat a stalled generation as stale anyway.
fn map_shm() -> Option<(std::os::fd::RawFd, *mut HxShmV1)> {
    let name = CString::new(HX_SHM_NAME).expect("SHM name contains no NUL");
    unsafe {
        libc::shm_unlink(name.as_ptr());
        let fd = libc::shm_open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);
        if fd < 0 {
            warn_shm_once(format!(
                "hands-shm: shm_open {HX_SHM_NAME} failed ({}), hand feed off",
                std::io::Error::last_os_error()
            ));
            return None;
        }
        if libc::ftruncate(fd, HX_SIZE as libc::off_t) != 0 {
            warn_shm_once(format!(
                "hands-shm: ftruncate failed ({}), hand feed off",
                std::io::Error::last_os_error()
            ));
            libc::close(fd);
            libc::shm_unlink(name.as_ptr());
            return None;
        }
        let base = libc::mmap(
            ptr::null_mut(),
            HX_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        if base == libc::MAP_FAILED {
            warn_shm_once(format!(
                "hands-shm: mmap failed ({}), hand feed off",
                std::io::Error::last_os_error()
            ));
            libc::close(fd);
            libc::shm_unlink(name.as_ptr());
            return None;
        }
        ptr::write_bytes(base as *mut u8, 0, HX_SIZE);
        // Prime the identity fields so a reader attaching before the first
        // publish sees magic/version instead of zeros.
        let shm = base as *mut HxShmV1;
        (&raw mut (*shm).magic).write(HX_MAGIC);
        (&raw mut (*shm).version).write(HX_VERSION);
        Some((fd, shm))
    }
}

fn copy_hand(dst: &mut [HxJoint; HX_JOINTS], src: &xr::HandJointLocations) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.pos = [s.pose.position.x, s.pose.position.y, s.pose.position.z];
        let o = s.pose.orientation;
        d.quat = [o.x, o.y, o.z, o.w];
    }
}

/// Publish one sample for both hands. `None` per hand = untracked (that
/// hand's joints are zeroed, its flag bit cleared). Always advances
/// `generation` on success — including all-invalid samples, which act as a
/// writer-alive heartbeat. Fail-silent: no-op when disabled or unmapped.
/// Never blocks on a reader (seqlock only).
pub fn publish(left: Option<&xr::HandJointLocations>, right: Option<&xr::HandJointLocations>) {
    if !hands_shm_enabled() {
        return;
    }
    let mut slot = SLOT.lock().unwrap_or_else(|p| p.into_inner());
    let Some(shm) = slot.ensure_locked() else {
        return;
    };
    let generation = GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    let now = now_mono_ns();
    let mut flags = 0u32;
    unsafe {
        // Seqlock: odd = writer inside.
        (*shm).seq.fetch_add(1, Ordering::AcqRel);
        (&raw mut (*shm).magic).write(HX_MAGIC);
        (&raw mut (*shm).version).write(HX_VERSION);
        match left {
            Some(joints) => {
                copy_hand(&mut (*shm).joints[0], joints);
                flags |= HX_LEFT_VALID;
            }
            None => (*shm).joints[0] = [HxJoint::default(); HX_JOINTS],
        }
        match right {
            Some(joints) => {
                copy_hand(&mut (*shm).joints[1], joints);
                flags |= HX_RIGHT_VALID;
            }
            None => (*shm).joints[1] = [HxJoint::default(); HX_JOINTS],
        }
        (&raw mut (*shm).flags).write(flags);
        (&raw mut (*shm).frame_mono_ns).write(now);
        (&raw mut (*shm).generation).write(generation);
        std::sync::atomic::fence(Ordering::Release);
        // Even = complete.
        (*shm).seq.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes tests that mutate process-global env vars.
    static ENV_GUARD: Mutex<()> = Mutex::new(());
    fn env_guard() -> MutexGuard<'static, ()> {
        ENV_GUARD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn save_env(key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
    fn restore_env(key: &str, saved: Option<String>) {
        match saved {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn gate_defaults_on_and_parses() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");

        unsafe { std::env::remove_var("XRIZER_HANDS_SHM") };
        assert!(hands_shm_enabled());
        for v in ["1", "true", "yes", "on", "  ON  "] {
            unsafe { std::env::set_var("XRIZER_HANDS_SHM", v) };
            assert!(hands_shm_enabled(), "{v:?}");
        }
        for v in ["0", "false", "no", "off", "  Off "] {
            unsafe { std::env::set_var("XRIZER_HANDS_SHM", v) };
            assert!(!hands_shm_enabled(), "{v:?}");
        }
        // Unparseable (incl. empty) warns once and stays ON.
        for v in ["bogus", ""] {
            unsafe { std::env::set_var("XRIZER_HANDS_SHM", v) };
            assert!(hands_shm_enabled(), "{v:?}");
        }

        restore_env("XRIZER_HANDS_SHM", saved);
    }

    fn sentinel_hand(mul: f32) -> xr::HandJointLocations {
        let mut joints = [xr::HandJointLocation::default(); HX_JOINTS];
        for (i, j) in joints.iter_mut().enumerate() {
            let f = i as f32 * mul;
            j.pose.position = xr::Vector3f {
                x: f,
                y: -f,
                z: 2.0 * f,
            };
            j.pose.orientation = xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            };
        }
        // One unmistakable joint.
        joints[0].pose.position = xr::Vector3f {
            x: 1234.5 * mul,
            y: -42.0,
            z: 7.25,
        };
        joints
    }

    /// Reader-discipline snapshot: retry while seq is odd or changed.
    /// Returns `(flags, frame_mono_ns, generation, joints)`.
    fn read_snapshot() -> Option<(u32, u64, u64, [[HxJoint; HX_JOINTS]; HX_HANDS])> {
        let name = CString::new(HX_SHM_NAME).unwrap();
        unsafe {
            let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
            if fd < 0 {
                return None;
            }
            let base = libc::mmap(
                ptr::null_mut(),
                HX_SIZE,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if base == libc::MAP_FAILED {
                return None;
            }
            let shm = base as *const HxShmV1;
            let mut out = None;
            for _ in 0..READ_TRIES {
                let a = (*shm).seq.load(Ordering::Acquire);
                if a & 1 != 0 {
                    continue;
                }
                let magic = (&raw const (*shm).magic).read();
                let version = (&raw const (*shm).version).read();
                let flags = (&raw const (*shm).flags).read();
                let t = (&raw const (*shm).frame_mono_ns).read();
                let generation = (&raw const (*shm).generation).read();
                let joints = (&raw const (*shm).joints).read();
                let b = (*shm).seq.load(Ordering::Acquire);
                if a != b {
                    continue;
                }
                assert_eq!((magic, version), (HX_MAGIC, HX_VERSION));
                out = Some((flags, t, generation, joints));
                break;
            }
            libc::munmap(base, HX_SIZE);
            out
        }
    }

    #[test]
    fn publish_roundtrip_left_only() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        let left = sentinel_hand(1.0);
        // Other tests may publish concurrently via frame_start_update; retry
        // until our own sample is observed.
        let mut seen = None;
        for _ in 0..100 {
            publish(Some(&left), None);
            if let Some((flags, _, generation, joints)) = read_snapshot() {
                let ok_flags = flags == HX_LEFT_VALID;
                let ok_j0 = (joints[0][0].pos[0] - 1234.5).abs() < 1e-6
                    && (joints[0][0].pos[1] + 42.0).abs() < 1e-6
                    && (joints[0][0].pos[2] - 7.25).abs() < 1e-6;
                let ok_quat = joints[0][0].quat == [0.0, 0.0, 0.0, 1.0];
                let ok_zeroed = joints[1]
                    .iter()
                    .all(|j| j.pos == [0.0; 3] && j.quat == [0.0; 4]);
                assert!(generation > 0, "generation must advance past 0");
                if ok_flags && ok_j0 && ok_quat && ok_zeroed {
                    seen = Some(generation);
                    break;
                }
            }
        }
        assert!(seen.is_some(), "never observed our own sample");

        restore_env("XRIZER_HANDS_SHM", saved);
    }
}
