//! Writer side of the `/xplane-hands-v2` hand-joint feed (xrizer -> shim).
//!
//! Byte-exact mirror of `shim/src/shm_v2.h` (magic `XPH2`, version 2,
//! 2120 bytes). v2 is forked from the FROZEN v1 contract in
//! `hands/src/shm.h` (never touch) and extends it with Gap 1
//! (tracking-space identity) + Gap 2 (joint linear velocities).
//!
//! v1-prefix readability: bytes [0, 1488) are offset-identical to v1
//! (magic+0 version+4 seq+8 flags+12 frame_mono_ns+16 generation+24
//! joints+32). All v2 additions (space words, velocities) are APPENDED
//! after the joints, so a v1-layout reader would decode the header and
//! joints correctly — but name, magic, and version all differ precisely so
//! mixed-version reads fail closed instead of silently succeeding.
//!
//! Size: header 32 B + joints 2*26*28 = 1456 B + space words 8 B +
//! velocities 2*26*vec3 = 624 B = 2120 B. The +632 B growth over v1 is one
//! linear-velocity vec3 per joint: differentiating positions costs a full
//! extra sample of latency and doubles reader state, while shipping the
//! runtime's `XrHandJointVelocitiesEXT` when valid is cheaper and exact.
//! Still a single small segment (fits one 4 KiB page).
//!
//! Writer discipline (mirrors `reprojex/src/publish.c`): unlink-then-create on
//! init so a fresh run never inherits a dead segment, then per publish bump
//! seq (odd = inside), write all fields, bump seq (even = complete). Readers
//! retry while seq is odd or changed across the read. Single writer; the
//! process-local [`SLOT`] mutex only serializes our own writer threads —
//! readers never take it, so the frame can never block on a reader.
//!
//! Coordinate frame: joints AND velocities are located relative to xrizer's
//! current tracking space (`SessionData::tracking_space()` — STAGE-adjusted
//! when Standing, LOCAL-adjusted when Seated), in meters, OpenXR
//! right-handed Y-up, in OpenXR joint order (`XrHandJointEXT` 0 = PALM ..
//! 25 = LITTLE_TIP). Velocities are m/s in that same frame. This is the
//! same frame the WaitGetPoses controller/HMD poses live in. Published
//! as-is (no OpenVR axis swap — that conversion is OpenVR-skeletal
//! specific); the shim consumes the values directly as its app frame. The
//! frame can change at runtime: `space_kind`/`space_gen` identify it (Gap 1)
//! because positions alone do not.
//!
//! Gap 1 — tracking-space identity (load-bearing): `space_kind` (0 =
//! unknown, 1 = local/Seated, 2 = stage/Standing) mirrors the STAGE/LOCAL
//! decision in `SessionData::tracking_space()`; `space_gen` bumps on every
//! `space_kind` change (0 = no sample yet). Both words are written on every
//! publish. Reader contract: cache the frame conversion derived from
//! `space_kind`, and on ANY `space_gen` change RE-DERIVE it and log loudly.
//!
//! Gap 2 — joint velocities: `vels[h][j]` is the runtime's
//! `XrHandJointVelocitiesEXT` linear velocity when valid, else zeros.
//! Validity is per-HAND, all-or-nothing: `HX_LEFT_VEL_VALID` /
//! `HX_RIGHT_VEL_VALID` are set only when the hand's joints are valid AND
//! every joint's `LINEAR_VALID` was set; otherwise the whole hand row is
//! zeroed and the bit cleared so the reader can fall back to
//! differentiating positions.
//!
//! Failure policy: fail silent. When the segment cannot be mapped (no
//! `/dev/shm`, read-only FS, …) publishing is skipped and a warning is logged
//! once per process; mapping is retried at most once per second. Kill switch:
//! `XRIZER_HANDS_SHM=0/false/no/off` disables publishing (default ON).

use openvr as vr;
use openxr as xr;
use std::ffi::CString;
use std::ptr;
use std::sync::{
    LazyLock, Mutex,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};

use log::{info, warn};

pub const HX_SHM_NAME: &str = "/xplane-hands-v2";
pub const HX_MAGIC: u32 = 0x58504832; // 'XPH2'
pub const HX_VERSION: u32 = 2;
pub const HX_HANDS: usize = 2;
pub const HX_JOINTS: usize = 26;
/// 32 B header + 2*26*28 B joints + 8 B space words + 2*26*12 B velocities.
pub const HX_SIZE: usize = 2120;
pub const HX_LEFT_VALID: u32 = 1 << 0;
pub const HX_RIGHT_VALID: u32 = 1 << 1;
/// Per-hand joint-velocity validity (Gap 2): set only when `vels[h]` holds
/// the runtime's linear velocities (every joint `LINEAR_VALID`). Clear =>
/// that hand row is zeros; the reader must differentiate positions instead.
pub const HX_LEFT_VEL_VALID: u32 = 1 << 2;
pub const HX_RIGHT_VEL_VALID: u32 = 1 << 3;
/// Tracking-space identity (Gap 1). Mirrors `SessionData::tracking_space()`.
pub const HX_SPACE_UNKNOWN: u32 = 0;
pub const HX_SPACE_LOCAL: u32 = 1;
pub const HX_SPACE_STAGE: u32 = 2;

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

/// Byte-exact mirror of `hx_shm_v2` in `shim/src/shm_v2.h`.
///
/// v1-prefix readability: `magic`..`joints` are offset-identical to the
/// frozen v1 layout; `space_kind`, `space_gen`, and `vels` are APPENDED
/// after the joints (never interleaved) so the v1 prefix stays readable.
#[repr(C)]
struct HxShmV2 {
    magic: u32,                               // +0
    version: u32,                             // +4
    seq: AtomicU32,                           // +8 seqlock (odd = writer inside)
    flags: u32,                               // +12 validity bits
    frame_mono_ns: u64,                       // +16 CLOCK_MONOTONIC of sample
    generation: u64,                          // +24 writer gen (0 = none yet)
    joints: [[HxJoint; HX_JOINTS]; HX_HANDS], // +32
    space_kind: u32,                          // +1488 0=unknown 1=local 2=stage
    space_gen: u32,                           // +1492 bumps on space_kind change
    vels: [[[f32; 3]; HX_JOINTS]; HX_HANDS],  // +1496 m/s, same frame
}

const _: () = assert!(size_of::<HxJoint>() == 28);
const _: () = assert!(size_of::<HxShmV2>() == HX_SIZE);
const _: () = assert!(size_of::<HxShmV2>() == 2120);
const _: () = assert!(size_of::<HxShmV2>() == 32 + 2 * 26 * 28 + 8 + 2 * 26 * 12);
const _: () = assert!(std::mem::offset_of!(HxShmV2, magic) == 0);
const _: () = assert!(std::mem::offset_of!(HxShmV2, version) == 4);
const _: () = assert!(std::mem::offset_of!(HxShmV2, seq) == 8);
const _: () = assert!(std::mem::offset_of!(HxShmV2, flags) == 12);
const _: () = assert!(std::mem::offset_of!(HxShmV2, frame_mono_ns) == 16);
const _: () = assert!(std::mem::offset_of!(HxShmV2, generation) == 24);
const _: () = assert!(std::mem::offset_of!(HxShmV2, joints) == 32);
/// v1-prefix readability, checked at compile time: the v2 additions start
/// exactly where the v1 joints end, so bytes [0, 1488) decode as v1.
const _: () = assert!(std::mem::offset_of!(HxShmV2, space_kind) == 32 + 2 * 26 * 28);
const _: () = assert!(std::mem::offset_of!(HxShmV2, space_gen) == 1492);
const _: () = assert!(std::mem::offset_of!(HxShmV2, space_kind) == 1488);
const _: () = assert!(std::mem::offset_of!(HxShmV2, space_gen) == 1492);
const _: () = assert!(std::mem::offset_of!(HxShmV2, vels) == 1496);

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
    ptr: *mut HxShmV2,
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

    fn ensure_locked(&mut self) -> Option<*mut HxShmV2> {
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
/// Gap 1: last published `space_kind` (starts impossible so the first
/// publish bumps) and the `space_gen` counter written alongside it.
static LAST_SPACE_KIND: AtomicU32 = AtomicU32::new(u32::MAX);
static SPACE_GEN: AtomicU32 = AtomicU32::new(0);
static WARNED_SHM: AtomicBool = AtomicBool::new(false);
/// Throttled-diagnostic counters (no new env vars; reuse log channels).
/// Counts silent `publish` early-outs so a dead feed is distinguishable.
static DISABLED_N: AtomicU64 = AtomicU64::new(0);
static UNMAPPED_N: AtomicU64 = AtomicU64::new(0);
/// One-shot "feed live" marker after priming; 1 Hz heartbeat throttle.
static FEED_LIVE_LOGGED: AtomicBool = AtomicBool::new(false);
static LAST_HB_NS: AtomicU64 = AtomicU64::new(0);
const HB_EVERY_NS: u64 = 1_000_000_000;

fn warn_shm_once(msg: String) {
    if !WARNED_SHM.swap(true, Ordering::Relaxed) {
        warn!("{msg}");
    }
}

/// Unlink-then-create + zero, mirroring `reprojex publish_ensure`. The fd is
/// held for the life of the process; the segment is intentionally not unlinked
/// on exit (no cdylib exit hook) — the next run recreates it fresh, and
/// readers must treat a stalled generation as stale anyway.
fn map_shm() -> Option<(std::os::fd::RawFd, *mut HxShmV2)> {
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
        let shm = base as *mut HxShmV2;
        (&raw mut (*shm).magic).write(HX_MAGIC);
        (&raw mut (*shm).version).write(HX_VERSION);
        if !FEED_LIVE_LOGGED.swap(true, Ordering::Relaxed) {
            info!(
                "hands-shm: feed live path={} size={} magic={:#010x}",
                HX_SHM_NAME, HX_SIZE, HX_MAGIC
            );
        }
        Some((fd, shm))
    }
}

/// Map the session's current tracking origin to the `space_kind` word
/// (Gap 1). Mirrors the STAGE/LOCAL decision in
/// `SessionData::tracking_space()`: Standing -> stage, Seated -> local,
/// anything else (RawAndUncalibrated) -> unknown.
pub fn space_kind_for_origin(origin: vr::ETrackingUniverseOrigin) -> u32 {
    match origin {
        vr::ETrackingUniverseOrigin::Seated => HX_SPACE_LOCAL,
        vr::ETrackingUniverseOrigin::Standing => HX_SPACE_STAGE,
        _ => HX_SPACE_UNKNOWN,
    }
}

/// One hand's sample for [`publish`]: joint locations plus optional joint
/// linear velocities (same order/count, `XrHandJointEXT` 0..26).
///
/// `velocities` should be the runtime's `XrHandJointVelocitiesEXT` output
/// (`Space::relate_hand_joints`). `publish` re-checks every joint's
/// `LINEAR_VALID`: if any is unset the whole hand row is zeroed and the
/// hand's vel bit cleared, so callers may pass velocities through
/// unchecked — but prefer `None` when the hand has no velocity source.
pub struct HandSample<'a> {
    pub locations: &'a xr::HandJointLocations,
    pub velocities: Option<&'a xr::HandJointVelocities>,
}

fn copy_hand(dst: &mut [HxJoint; HX_JOINTS], src: &xr::HandJointLocations) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.pos = [s.pose.position.x, s.pose.position.y, s.pose.position.z];
        let o = s.pose.orientation;
        d.quat = [o.x, o.y, o.z, o.w];
    }
}

/// Copy linear velocities (m/s, same frame as positions). Returns true iff
/// every joint's `LINEAR_VALID` was set. Any gap -> the caller must zero
/// the row and clear the hand's vel bit so the reader falls back to
/// differentiating positions.
fn copy_vels(dst: &mut [[f32; 3]; HX_JOINTS], src: &xr::HandJointVelocities) -> bool {
    if !src
        .iter()
        .all(|v| v.velocity_flags.contains(xr::SpaceVelocityFlags::LINEAR_VALID))
    {
        return false;
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let l = s.linear_velocity;
        *d = [l.x, l.y, l.z];
    }
    true
}

/// Publish one sample for both hands. `None` per hand = untracked (that
/// hand's joints AND velocities are zeroed, its flag bits cleared). Always
/// advances `generation` on success — including all-invalid samples, which
/// act as a writer-alive heartbeat. Fail-silent: no-op when disabled or
/// unmapped. Never blocks on a reader (seqlock only).
///
/// `space_kind` (see [`space_kind_for_origin`]) is written every publish;
/// `space_gen` bumps on every `space_kind` change (first publish always
/// bumps: the last-kind tracker starts at an impossible value). Both words
/// are part of the seqlock, so a reader always sees a consistent
/// (kind, gen, joints, vels) snapshot.
///
/// Returns `(generation, flags, left_wrist, right_wrist)` as written (zeros
/// when disabled/unmapped) so the per-second `hands locate` line in
/// `input.rs` can prove real data flows (wave test: successive lines must
/// show wrist delta > 0.01 m when hands move).
pub fn publish(
    left: Option<HandSample<'_>>,
    right: Option<HandSample<'_>>,
    space_kind: u32,
) -> (u64, u32, [f32; 3], [f32; 3]) {
    if !hands_shm_enabled() {
        let n = DISABLED_N.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 {
            warn_shm_once(
                "hands-shm: disabled via XRIZER_HANDS_SHM, hand feed off".to_string(),
            );
        } else if n.is_multiple_of(3600) {
            info!("hands-shm: disabled via XRIZER_HANDS_SHM, hand feed off n={n}");
        }
        return (0, 0, [0.0; 3], [0.0; 3]);
    }
    let mut slot = SLOT.lock().unwrap_or_else(|p| p.into_inner());
    let Some(shm) = slot.ensure_locked() else {
        let n = UNMAPPED_N.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 {
            warn_shm_once(format!("hands-shm: shm unmapped n={n}, hand feed off"));
        } else if n.is_multiple_of(3600) {
            info!("hands-shm: shm unmapped n={n}, hand feed off");
        }
        return (0, 0, [0.0; 3], [0.0; 3]);
    };
    let generation = GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    // Gap 1: bump space_gen on every space_kind change. The swap runs on
    // every publish so (kind, gen) are always a consistent pair under the
    // seqlock below; LAST_SPACE_KIND starts impossible so the first publish
    // bumps (gen 1, 0 = no sample yet).
    let prev_kind = LAST_SPACE_KIND.swap(space_kind, Ordering::Relaxed);
    let space_gen = if space_kind != prev_kind {
        SPACE_GEN.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        SPACE_GEN.load(Ordering::Relaxed)
    };
    let now = now_mono_ns();
    let mut flags = 0u32;
    // Captured while the slot is locked so the heartbeat below reports
    // exactly what was written this sample.
    let (l0, r0) = unsafe {
        // Seqlock: odd = writer inside.
        (*shm).seq.fetch_add(1, Ordering::AcqRel);
        (&raw mut (*shm).magic).write(HX_MAGIC);
        (&raw mut (*shm).version).write(HX_VERSION);
        match left {
            Some(sample) => {
                copy_hand(&mut (*shm).joints[0], sample.locations);
                flags |= HX_LEFT_VALID;
                let vel_ok = match sample.velocities {
                    Some(v) => copy_vels(&mut (*shm).vels[0], v),
                    None => false,
                };
                if vel_ok {
                    flags |= HX_LEFT_VEL_VALID;
                } else {
                    (*shm).vels[0] = [[0.0; 3]; HX_JOINTS];
                }
            }
            None => {
                (*shm).joints[0] = [HxJoint::default(); HX_JOINTS];
                (*shm).vels[0] = [[0.0; 3]; HX_JOINTS];
            }
        }
        match right {
            Some(sample) => {
                copy_hand(&mut (*shm).joints[1], sample.locations);
                flags |= HX_RIGHT_VALID;
                let vel_ok = match sample.velocities {
                    Some(v) => copy_vels(&mut (*shm).vels[1], v),
                    None => false,
                };
                if vel_ok {
                    flags |= HX_RIGHT_VEL_VALID;
                } else {
                    (*shm).vels[1] = [[0.0; 3]; HX_JOINTS];
                }
            }
            None => {
                (*shm).joints[1] = [HxJoint::default(); HX_JOINTS];
                (*shm).vels[1] = [[0.0; 3]; HX_JOINTS];
            }
        }
        (&raw mut (*shm).flags).write(flags);
        (&raw mut (*shm).frame_mono_ns).write(now);
        (&raw mut (*shm).generation).write(generation);
        (&raw mut (*shm).space_kind).write(space_kind);
        (&raw mut (*shm).space_gen).write(space_gen);
        std::sync::atomic::fence(Ordering::Release);
        // Even = complete.
        (*shm).seq.fetch_add(1, Ordering::AcqRel);
        (
            (&raw const (*shm).joints[0][0].pos).read(),
            (&raw const (*shm).joints[1][0].pos).read(),
        )
    };
    // 1 Hz heartbeat while the slot is still locked (same wrapping-sub due
    // check as `log_hands_locate_throttled` in input.rs).
    let last_hb = LAST_HB_NS.load(Ordering::Relaxed);
    if now.wrapping_sub(last_hb) >= HB_EVERY_NS {
        LAST_HB_NS.store(now, Ordering::Relaxed);
        info!(
            "hands-shm: hb gen={} flags={:#06b} sk={} sg={} L0=({:.3},{:.3},{:.3}) mono_ns={}",
            generation, flags, space_kind, space_gen, l0[0], l0[1], l0[2], now
        );
    }
    (generation, flags, l0, r0)
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

    /// Sentinel velocities: distinctive linear values; when `all_valid` is
    /// false joint 3 reports no `LINEAR_VALID` (rest are valid).
    fn sentinel_vels(mul: f32, all_valid: bool) -> xr::HandJointVelocities {
        std::array::from_fn(|i| xr::HandJointVelocity {
            velocity_flags: if all_valid || i != 3 {
                xr::SpaceVelocityFlags::LINEAR_VALID
            } else {
                xr::SpaceVelocityFlags::EMPTY
            },
            linear_velocity: xr::Vector3f {
                x: 100.0 * mul + i as f32,
                y: -mul,
                z: 0.5 * mul,
            },
            ..Default::default()
        })
    }

    struct Snapshot {
        flags: u32,
        frame_mono_ns: u64,
        generation: u64,
        joints: [[HxJoint; HX_JOINTS]; HX_HANDS],
        space_kind: u32,
        space_gen: u32,
        vels: [[[f32; 3]; HX_JOINTS]; HX_HANDS],
    }

    /// Reader-discipline snapshot: retry while seq is odd or changed.
    fn read_snapshot() -> Option<Snapshot> {
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
            let shm = base as *const HxShmV2;
            let mut out = None;
            for _ in 0..READ_TRIES {
                let a = (*shm).seq.load(Ordering::Acquire);
                if a & 1 != 0 {
                    continue;
                }
                let magic = (&raw const (*shm).magic).read();
                let version = (&raw const (*shm).version).read();
                let snap = Snapshot {
                    flags: (&raw const (*shm).flags).read(),
                    frame_mono_ns: (&raw const (*shm).frame_mono_ns).read(),
                    generation: (&raw const (*shm).generation).read(),
                    joints: (&raw const (*shm).joints).read(),
                    space_kind: (&raw const (*shm).space_kind).read(),
                    space_gen: (&raw const (*shm).space_gen).read(),
                    vels: (&raw const (*shm).vels).read(),
                };
                let b = (*shm).seq.load(Ordering::Acquire);
                if a != b {
                    continue;
                }
                assert_eq!((magic, version), (HX_MAGIC, HX_VERSION));
                out = Some(snap);
                break;
            }
            libc::munmap(base, HX_SIZE);
            out
        }
    }

    fn sample<'a>(
        locations: &'a xr::HandJointLocations,
        velocities: Option<&'a xr::HandJointVelocities>,
    ) -> HandSample<'a> {
        HandSample {
            locations,
            velocities,
        }
    }

    fn joint0_ok(joints: &[[HxJoint; HX_JOINTS]; HX_HANDS], mul: f32) -> bool {
        (joints[0][0].pos[0] - 1234.5 * mul).abs() < 1e-6
            && (joints[0][0].pos[1] + 42.0).abs() < 1e-6
            && (joints[0][0].pos[2] - 7.25).abs() < 1e-6
            && joints[0][0].quat == [0.0, 0.0, 0.0, 1.0]
    }

    fn hand_zeroed(joints: &[HxJoint; HX_JOINTS], vels: &[[f32; 3]; HX_JOINTS]) -> bool {
        joints
            .iter()
            .all(|j| j.pos == [0.0; 3] && j.quat == [0.0; 4])
            && vels.iter().all(|v| *v == [0.0; 3])
    }

    fn vels_ok(vels: &[[[f32; 3]; HX_JOINTS]; HX_HANDS], hand: usize, mul: f32) -> bool {
        vels[hand].iter().enumerate().all(|(i, v)| {
            (v[0] - (100.0 * mul + i as f32)).abs() < 1e-6
                && (v[1] + 1.0 * mul).abs() < 1e-6
                && (v[2] - 0.5 * mul).abs() < 1e-6
        })
    }

    #[test]
    fn space_kind_mapping() {
        use vr::ETrackingUniverseOrigin as Origin;
        assert_eq!(space_kind_for_origin(Origin::Seated), HX_SPACE_LOCAL);
        assert_eq!(space_kind_for_origin(Origin::Standing), HX_SPACE_STAGE);
        assert_eq!(
            space_kind_for_origin(Origin::RawAndUncalibrated),
            HX_SPACE_UNKNOWN
        );
    }

    #[test]
    fn publish_roundtrip_left_only() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        let left = sentinel_hand(1.0);
        let left_vel = sentinel_vels(1.0, true);
        // Other tests may publish concurrently via frame_start_update; retry
        // until our own sample is observed.
        let mut seen = None;
        for _ in 0..200 {
            publish(Some(sample(&left, Some(&left_vel))), None, HX_SPACE_LOCAL);
            if let Some(snap) = read_snapshot() {
                let ok_flags = snap.flags == (HX_LEFT_VALID | HX_LEFT_VEL_VALID);
                let ok_kind = snap.space_kind == HX_SPACE_LOCAL && snap.space_gen > 0;
                let ok_vel = vels_ok(&snap.vels, 0, 1.0);
                let ok_zeroed = hand_zeroed(&snap.joints[1], &snap.vels[1]);
                assert!(snap.generation > 0, "generation must advance past 0");
                assert!(snap.frame_mono_ns > 0, "frame time must be stamped");
                if ok_flags && ok_kind && joint0_ok(&snap.joints, 1.0) && ok_vel && ok_zeroed
                {
                    seen = Some(snap.generation);
                    break;
                }
            }
        }
        assert!(seen.is_some(), "never observed our own sample");

        restore_env("XRIZER_HANDS_SHM", saved);
    }

    #[test]
    fn publish_vel_invalid_zeroes_and_clears_bit() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        // Joint 3 lacks LINEAR_VALID: the whole hand row must be zeroed and
        // the vel bit cleared, while joints still publish.
        let left = sentinel_hand(3.0);
        let left_vel = sentinel_vels(3.0, false);
        let mut seen = false;
        for _ in 0..200 {
            publish(Some(sample(&left, Some(&left_vel))), None, HX_SPACE_STAGE);
            if let Some(snap) = read_snapshot() {
                let ok_kind = snap.space_kind == HX_SPACE_STAGE;
                if snap.flags == HX_LEFT_VALID
                    && ok_kind
                    && joint0_ok(&snap.joints, 3.0)
                    && snap.vels[0].iter().all(|v| *v == [0.0; 3])
                {
                    seen = true;
                    break;
                }
            }
        }
        assert!(seen, "never observed invalid-vel sample zeroed");

        restore_env("XRIZER_HANDS_SHM", saved);
    }

    #[test]
    fn publish_vel_none_zeroes_and_clears_bit() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        // No velocity source: row zeroed, vel bit clear, joints live.
        let left = sentinel_hand(5.0);
        let mut seen = false;
        for _ in 0..200 {
            publish(Some(sample(&left, None)), None, HX_SPACE_LOCAL);
            if let Some(snap) = read_snapshot()
                && snap.flags == HX_LEFT_VALID
                && snap.space_kind == HX_SPACE_LOCAL
                && joint0_ok(&snap.joints, 5.0)
                && snap.vels[0].iter().all(|v| *v == [0.0; 3])
            {
                seen = true;
                break;
            }
        }
        assert!(seen, "never observed vel-less sample zeroed");

        restore_env("XRIZER_HANDS_SHM", saved);
    }

    #[test]
    fn publish_space_gen_bumps_on_change() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        // Observe our own sample under two different space kinds (identified
        // by distinct sentinel joints). space_gen is a monotonic counter, so
        // the second sample's gen is strictly greater regardless of
        // concurrent publishers: our own publish either bumps it itself or a
        // concurrent kind-change already did.
        let first = sentinel_hand(7.0);
        let second = sentinel_hand(8.0);
        let mut first_gen = None;
        for _ in 0..200 {
            publish(Some(sample(&first, None)), None, HX_SPACE_LOCAL);
            if let Some(snap) = read_snapshot()
                && snap.flags == HX_LEFT_VALID
                && snap.space_kind == HX_SPACE_LOCAL
                && joint0_ok(&snap.joints, 7.0)
            {
                first_gen = Some(snap.space_gen);
                break;
            }
        }
        let first_gen = first_gen.expect("never observed first space sample");
        assert!(first_gen > 0, "space_gen must advance past 0");

        let mut second_gen = None;
        for _ in 0..200 {
            publish(Some(sample(&second, None)), None, HX_SPACE_STAGE);
            if let Some(snap) = read_snapshot()
                && snap.flags == HX_LEFT_VALID
                && snap.space_kind == HX_SPACE_STAGE
                && joint0_ok(&snap.joints, 8.0)
            {
                second_gen = Some(snap.space_gen);
                break;
            }
        }
        let second_gen = second_gen.expect("never observed second space sample");
        assert!(
            second_gen > first_gen,
            "space_gen must bump on space_kind change ({second_gen} <= {first_gen})"
        );

        restore_env("XRIZER_HANDS_SHM", saved);
    }

    #[test]
    fn v1_prefix_readable_at_v1_offsets() {
        let _g = env_guard();
        let saved = save_env("XRIZER_HANDS_SHM");
        unsafe { std::env::set_var("XRIZER_HANDS_SHM", "1") };

        // Decode the segment with the FROZEN v1 field offsets to prove the
        // v1 prefix survived: magic+0 version+4 flags+12 joints+32.
        let left = sentinel_hand(9.0);
        let mut seen = false;
        for _ in 0..200 {
            publish(Some(sample(&left, None)), None, HX_SPACE_STAGE);
            let name = CString::new(HX_SHM_NAME).unwrap();
            unsafe {
                let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
                if fd < 0 {
                    continue;
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
                    continue;
                }
                let bytes =
                    std::slice::from_raw_parts(base as *const u8, HX_SIZE);
                let u = |off: usize| {
                    u32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap())
                };
                let f = |off: usize| {
                    f32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap())
                };
                // Seqlock a consistent read the same way a v1 reader would.
                // (Audit fix 2026-11-02: seq lives at +8, not +0 — the old
                // cast read magic, which is constant-even, making the tear
                // check vacuous.)
                let seq_ptr = unsafe { base.byte_add(8) } as *const AtomicU32;
                let a = (*seq_ptr).load(Ordering::Acquire);
                let (magic, version, flags, jx, jy, jz) =
                    (u(0), u(4), u(12), f(32), f(36), f(40));
                let b = (*seq_ptr).load(Ordering::Acquire);
                libc::munmap(base, HX_SIZE);
                if a & 1 != 0 || a != b {
                    continue;
                }
                // Low 2 flag bits are the v1 validity bits (vel bits are new).
                if magic == HX_MAGIC
                    && version == HX_VERSION
                    && flags & 0x3 == HX_LEFT_VALID
                    && (jx - 1234.5 * 9.0).abs() < 1e-6
                    && (jy + 42.0).abs() < 1e-6
                    && (jz - 7.25).abs() < 1e-6
                {
                    seen = true;
                    break;
                }
            }
        }
        assert!(seen, "v1 prefix not decodable at v1 offsets");

        restore_env("XRIZER_HANDS_SHM", saved);
    }
}
