//! SHM depth handoff reader (X-Plane layer -> xrizer).
//!
//! Transport is a POSIX SHM latest-slot segment (`/xplane-depth-v1` by default,
//! overridable via `XP_DEPTH_SHM_NAME` for tests). Layout is little-endian,
//! all offsets fixed:
//!
//! ```text
//! +0  u32 magic       0x58504450 ('XPDP')
//! +4  u32 version     1 or 2
//! +8  u32 seq         seqlock: odd = writer inside
//! +12 u32 w, +16 u32 h
//! +20 i32 format      Vulkan format enum (126 = D32F, 130 = D32S8)
//! +24 u32 eye_layout  1 = side-by-side (left half eye0, right half eye1)
//! +32 u64 present_no  (FD: latest signaled generation)
//! +40 u64 time_mono_ns (CLOCK_MONOTONIC when copy completed)
//! +48 f32 min_depth, +52 f32 max_depth
//! +56 pad[8] (unnamed)
//! +64 u32 mode        0 = PIXELS, 1 = FD              (v2 only)
//! +68 u32 setup_epoch                                  (v2 only)
//! +72 u64 consumed_gen (written by xrizer: highest fully-copied generation)
//! +80 i32 mem_fds[3], +92 i32 sem_fds[3]               (v2 only, same process)
//! +104 f32 near_z, +108 f32 far_z (0 = unknown -> env)
//! +112 u32 reserved[4]
//! +128 f32[w*h] row-major pixels (PIXELS mode only; v1: at +64)
//! ```
//!
//! v1 (64-byte header) is tolerated as PIXELS-at-+64.
//!
//! No new crates: `libc` only. Absent/stale/mismatched SHM always yields
//! `None` so callers fall back to plain (color-only) views.

use log::{debug, info};
use std::ffi::CString;

pub const SHM_MAGIC: u32 = 0x5850_4450; // 'XPDP'
pub const SHM_VERSION: u32 = 1;
pub const SHM_VERSION_V2: u32 = 2;
pub const EYE_SIDEBYSIDE: u32 = 1;
pub const MODE_PIXELS: u32 = 0;
pub const MODE_FD: u32 = 1;
/// Vulkan `VK_FORMAT_D32_SFLOAT`.
pub const FORMAT_D32F: i32 = 126;
/// Vulkan `VK_FORMAT_D32_SFLOAT_S8_UINT`.
pub const FORMAT_D32S8: i32 = 130;
pub const HEADER_SIZE: usize = 64;
pub const HEADER_SIZE_V2: usize = 128;
/// Byte offset of `consumed_gen`: written by xrizer, 8-aligned.
pub const CONSUMED_GEN_OFF: usize = 72;

pub const DEFAULT_SHM_NAME: &str = "/xplane-depth-v1";
pub const GL_DEPTH_COMPONENT32F: u32 = 0x8CAC;
pub const GL_DEPTH32F_STENCIL8: u32 = 0x8CAD;

/// Mirror of the C `xd_shm_header`. Must stay 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct XdShmHeader {
    pub magic: u32,         // +0
    pub version: u32,       // +4
    pub seq: u32,           // +8
    pub w: u32,             // +12
    pub h: u32,             // +16
    pub format: i32,        // +20
    pub eye_layout: u32,    // +24
    pub present_no: u64,    // +32
    pub time_mono_ns: u64,  // +40
    pub min_depth: f32,     // +48
    pub max_depth: f32,     // +52
    pub reserved: [u32; 2], // +56,+60
}

#[cfg(test)]
const _: () = assert!(std::mem::size_of::<XdShmHeader>() == HEADER_SIZE);

/// Mirror of the C v2 `xd_shm_header`. Must stay 128 bytes with the
/// exact offsets above (`mode` at +64, `consumed_gen` at +72, ...).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct XdShmHeaderV2 {
    pub magic: u32,         // +0
    pub version: u32,       // +4
    pub seq: u32,           // +8
    pub w: u32,             // +12
    pub h: u32,             // +16
    pub format: i32,        // +20
    pub eye_layout: u32,    // +24
    pub present_no: u64,    // +32 latest signaled generation (FD)
    pub time_mono_ns: u64,  // +40
    pub min_depth: f32,     // +48
    pub max_depth: f32,     // +52
    pub _pad56: [u32; 2],   // +56,+60 unnamed
    pub mode: u32,          // +64 0 = PIXELS, 1 = FD
    pub setup_epoch: u32,   // +68
    pub consumed_gen: u64,  // +72 written by xrizer
    pub mem_fds: [i32; 3],  // +80
    pub sem_fds: [i32; 3],  // +92
    pub near_z: f32,        // +104 0 = unknown -> env
    pub far_z: f32,         // +108 0 = unknown -> env
    pub reserved: [u32; 4], // +112
}

#[cfg(test)]
const _: () = {
    assert!(std::mem::size_of::<XdShmHeaderV2>() == HEADER_SIZE_V2);
    assert!(std::mem::offset_of!(XdShmHeaderV2, mode) == 64);
    assert!(std::mem::offset_of!(XdShmHeaderV2, setup_epoch) == 68);
    assert!(std::mem::offset_of!(XdShmHeaderV2, consumed_gen) == CONSUMED_GEN_OFF);
    assert!(std::mem::offset_of!(XdShmHeaderV2, mem_fds) == 80);
    assert!(std::mem::offset_of!(XdShmHeaderV2, sem_fds) == 92);
    assert!(std::mem::offset_of!(XdShmHeaderV2, near_z) == 104);
    assert!(std::mem::offset_of!(XdShmHeaderV2, far_z) == 108);
};

/// Owned copy of one SHM publish.
#[derive(Debug)]
pub struct DepthFrame {
    pub w: u32,
    pub h: u32,
    pub present_no: u64,
    pub time_ns: u64,
    pub pixels: Vec<f32>,
    pub vk_format: i32,
    pub near_z: f32,
    pub far_z: f32,
}

impl DepthFrame {
    pub fn eye_w(&self) -> u32 {
        self.w / 2
    }
}

/// Snapshot of the v1/v2 control header. FD transport reads this;
/// the pixel path uses it for mode dispatch.
#[derive(Debug, Clone)]
pub struct DepthControl {
    pub version: u32,
    pub mode: u32,
    pub setup_epoch: u32,
    pub w: u32,
    pub h: u32,
    pub format: i32,
    pub eye_layout: u32,
    /// Latest signaled generation (`present_no`).
    pub generation: u64,
    pub time_ns: u64,
    pub mem_fds: [i32; 3],
    pub sem_fds: [i32; 3],
    pub near_z: f32,
    pub far_z: f32,
    /// Highest fully-copied generation, as last observed.
    pub consumed_gen: u64,
    pub seg_len: usize,
}

/// Reader handle. Stateless across frames (open/mmap per read) so producer
/// `munmap/shm_unlink/recreate` cycles on frame-graph rebuilds are picked up
/// automatically with no stale fd.
pub struct DepthFeed {
    name: CString,
}

struct MappedSeg {
    fd: libc::c_int,
    ptr: *mut libc::c_void,
    len: usize,
}

impl Drop for MappedSeg {
    fn drop(&mut self) {
        if self.ptr != libc::MAP_FAILED && !self.ptr.is_null() {
            unsafe {
                libc::munmap(self.ptr, self.len);
            }
        }
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// Open + mmap the current segment. A fresh mapping every call picks up
/// producer recreates (epoch or size changes) with no stale fd.
fn map_segment(name: &CString, write: bool) -> Option<MappedSeg> {
    let oflag = (if write { libc::O_RDWR } else { libc::O_RDONLY }) | libc::O_CLOEXEC;
    let fd = unsafe { libc::shm_open(name.as_ptr(), oflag, 0o600) };
    if fd < 0 {
        debug!("depth feed: shm_open failed (absent)");
        return None;
    }
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        unsafe { libc::close(fd) };
        return None;
    }
    let map_len = stat.st_size as usize;
    if map_len == 0 {
        unsafe { libc::close(fd) };
        return None;
    }
    let prot = libc::PROT_READ | (if write { libc::PROT_WRITE } else { 0 });
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), map_len, prot, libc::MAP_SHARED, fd, 0) };
    if ptr == libc::MAP_FAILED {
        unsafe { libc::close(fd) };
        return None;
    }
    Some(MappedSeg {
        fd,
        ptr,
        len: map_len,
    })
}

impl DepthFeed {
    /// `None` when the `XRIZER_DEPTH_SHM=1` master switch is off.
    /// Logs the resolved SHM path once so cross-repo mismatches show up fast.
    pub fn open() -> Option<Self> {
        if !depth_master_enabled() {
            return None;
        }
        let name = shm_name();
        info!("depth feed: SHM path {name} (master on)");
        Some(Self {
            name: CString::new(name).expect("SHM name contains NUL"),
        })
    }

    /// Test seam: bypass the master switch.
    #[cfg(test)]
    fn open_for_test(name: &str) -> Self {
        Self {
            name: CString::new(name).expect("SHM name contains NUL"),
        }
    }

    /// Latest depth with `time_ns <= now_mono_ns` and age within `max_age_ms`.
    /// Returns `None` on absent/stale/future/mismatched data (plain views).
    pub fn latest_for(&self, now_mono_ns: u64, max_age_ms: u64) -> Option<DepthFrame> {
        // Open current segment read-only each call: handles producer recreate.
        let map = map_segment(&self.name, false)?;
        let (ptr, map_len) = (map.ptr, map.len);
        if map_len < HEADER_SIZE {
            return None;
        }

        // Seqlock read: retry while odd or changed. Bounded; writer publishes
        // rarely (dump ticks only) so contention is essentially impossible.
        for _ in 0..4 {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            if unsafe { read_u32(ptr, 0) } != SHM_MAGIC {
                return None;
            }
            // Parsed pixel publish, version-dispatched. v1 = PIXELS at +64;
            // v2 must be PIXELS mode (FD is served by read_control).
            struct Pix {
                w: u32,
                h: u32,
                format: i32,
                layout: u32,
                present_no: u64,
                time_ns: u64,
                near_z: f32,
                far_z: f32,
                pix_off: usize,
            }
            let p: Pix = match unsafe { read_u32(ptr, 4) } {
                SHM_VERSION => {
                    let hdr = unsafe { read_header(ptr) };
                    if !header_plausible(&hdr, map_len) {
                        return None;
                    }
                    Pix {
                        w: hdr.w,
                        h: hdr.h,
                        format: hdr.format,
                        layout: hdr.eye_layout,
                        present_no: hdr.present_no,
                        time_ns: hdr.time_mono_ns,
                        near_z: 0.0,
                        far_z: 0.0,
                        pix_off: HEADER_SIZE,
                    }
                }
                SHM_VERSION_V2 => {
                    if map_len < HEADER_SIZE_V2 {
                        return None;
                    }
                    let hdr = unsafe { read_header_v2(ptr) };
                    if hdr.mode != MODE_PIXELS {
                        return None;
                    }
                    if !seg_size_ok(hdr.w, hdr.h, HEADER_SIZE_V2, map_len) {
                        return None;
                    }
                    Pix {
                        w: hdr.w,
                        h: hdr.h,
                        format: hdr.format,
                        layout: hdr.eye_layout,
                        present_no: hdr.present_no,
                        time_ns: hdr.time_mono_ns,
                        near_z: hdr.near_z,
                        far_z: hdr.far_z,
                        pix_off: HEADER_SIZE_V2,
                    }
                }
                _ => return None,
            };
            // Copy header + pixels under seq1.
            let npix = p.w as usize * p.h as usize;
            let mut pixels = vec![0f32; npix];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (ptr as *const u8).add(p.pix_off) as *const f32,
                    pixels.as_mut_ptr(),
                    npix,
                );
            }
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            // Accept only magic+version, float32, side-by-side.
            if p.format != FORMAT_D32F || p.layout != EYE_SIDEBYSIDE {
                return None;
            }
            // Staleness: future data or older than max age.
            // (PIXELS only; the FD path uses the timeline instead.)
            if p.time_ns > now_mono_ns {
                return None;
            }
            let max_age_ns = max_age_ms.saturating_mul(1_000_000);
            if now_mono_ns.saturating_sub(p.time_ns) > max_age_ns {
                return None;
            }
            return Some(DepthFrame {
                w: p.w,
                h: p.h,
                present_no: p.present_no,
                time_ns: p.time_ns,
                pixels,
                vk_format: p.format,
                near_z: p.near_z,
                far_z: p.far_z,
            });
        }
        None
    }

    /// Seqlock-read of the v1/v2 control header. No pixel copy.
    pub fn read_control(&self) -> Option<DepthControl> {
        let map = map_segment(&self.name, false)?;
        let (ptr, map_len) = (map.ptr, map.len);
        if map_len < 16 {
            return None;
        }
        for _ in 0..4 {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            if unsafe { read_u32(ptr, 0) } != SHM_MAGIC {
                return None;
            }
            let ctl = match unsafe { read_u32(ptr, 4) } {
                SHM_VERSION => {
                    if map_len < HEADER_SIZE {
                        return None;
                    }
                    let h = unsafe { read_header(ptr) };
                    DepthControl {
                        version: SHM_VERSION,
                        mode: MODE_PIXELS,
                        setup_epoch: 0,
                        w: h.w,
                        h: h.h,
                        format: h.format,
                        eye_layout: h.eye_layout,
                        generation: h.present_no,
                        time_ns: h.time_mono_ns,
                        mem_fds: [-1; 3],
                        sem_fds: [-1; 3],
                        near_z: 0.0,
                        far_z: 0.0,
                        consumed_gen: 0,
                        seg_len: map_len,
                    }
                }
                SHM_VERSION_V2 => {
                    if map_len < HEADER_SIZE_V2 {
                        return None;
                    }
                    let h = unsafe { read_header_v2(ptr) };
                    DepthControl {
                        version: SHM_VERSION_V2,
                        mode: h.mode,
                        setup_epoch: h.setup_epoch,
                        w: h.w,
                        h: h.h,
                        format: h.format,
                        eye_layout: h.eye_layout,
                        generation: h.present_no,
                        time_ns: h.time_mono_ns,
                        mem_fds: h.mem_fds,
                        sem_fds: h.sem_fds,
                        near_z: h.near_z,
                        far_z: h.far_z,
                        consumed_gen: h.consumed_gen,
                        seg_len: map_len,
                    }
                }
                _ => return None,
            };
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            return Some(ctl);
        }
        None
    }

    /// Record the highest fully-copied generation. Single aligned u64 store;
    /// the producer preserves it across header rewrites, and a lost update
    /// only skips a slot (benign). Always stores (bounded even-seq spin
    /// first): a missed ack stalls the producer ring.
    pub fn store_consumed(&self, generation: u64) -> bool {
        let Some(map) = map_segment(&self.name, true) else {
            return false;
        };
        if map.len < CONSUMED_GEN_OFF + 8 {
            return false;
        }
        for _ in 0..100 {
            if unsafe { read_seq(map.ptr) } & 1 == 0 {
                break;
            }
            std::hint::spin_loop();
        }
        unsafe {
            (map.ptr as *mut u8)
                .add(CONSUMED_GEN_OFF)
                .cast::<u64>()
                .write(generation)
        };
        true
    }
}

unsafe fn read_seq(base: *mut libc::c_void) -> u32 {
    unsafe { (base as *const u8).add(8).cast::<u32>().read_volatile() }
}

unsafe fn read_u32(base: *mut libc::c_void, off: usize) -> u32 {
    unsafe { (base as *const u8).add(off).cast::<u32>().read() }
}

unsafe fn read_header(base: *mut libc::c_void) -> XdShmHeader {
    // Field-by-field volatile read is overkill; the seqlock re-check covers
    // torn reads. Plain copy out of the mapping.
    unsafe { (base as *const XdShmHeader).read_unaligned() }
}

unsafe fn read_header_v2(base: *mut libc::c_void) -> XdShmHeaderV2 {
    unsafe { (base as *const XdShmHeaderV2).read_unaligned() }
}

fn seg_size_ok(w: u32, h: u32, pix_base: usize, map_len: usize) -> bool {
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return false;
    }
    if w % 2 != 0 {
        return false;
    }
    let want = pix_base + w as usize * h as usize * 4;
    // Require exact size: producer recreates the segment on dims change.
    want == map_len
}

fn header_plausible(hdr: &XdShmHeader, map_len: usize) -> bool {
    seg_size_ok(hdr.w, hdr.h, HEADER_SIZE, map_len)
}

/// Header near/far win when both are positive and distinct; otherwise the
/// `XRIZER_DEPTH_NEAR` / `XRIZER_DEPTH_FAR` env pair (tunable, no rebuild).
pub fn resolve_near_far(hdr_near: f32, hdr_far: f32) -> (f32, f32) {
    if hdr_near > 0.0 && hdr_far > 0.0 && hdr_near != hdr_far {
        (hdr_near, hdr_far)
    } else {
        near_far()
    }
}

/// Master switch, default off.
pub fn depth_master_enabled() -> bool {
    std::env::var("XRIZER_DEPTH_SHM")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Shared path constant, test-overridable. Single source of truth with the
/// C producer/reader (`test_shm_read`).
pub fn shm_name() -> String {
    std::env::var("XP_DEPTH_SHM_NAME").unwrap_or_else(|_| DEFAULT_SHM_NAME.to_string())
}

pub fn max_age_ms() -> u64 {
    std::env::var("XRIZER_DEPTH_MAX_AGE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

pub fn near_far() -> (f32, f32) {
    let near = std::env::var("XRIZER_DEPTH_NEAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let far = std::env::var("XRIZER_DEPTH_FAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30000.0);
    (near, far)
}

/// `CLOCK_MONOTONIC` in ns. Same clock the producer stamps `time_mono_ns`
/// with, same machine.
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

#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

/// Serializes tests that mutate process-global env vars (shared by this
/// module's tests and the compositor gate tests).
#[cfg(test)]
static ENV_GUARD: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn env_guard() -> MutexGuard<'static, ()> {
    ENV_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Test-only env mutation. Callers must hold [`env_guard`].
#[cfg(test)]
pub(crate) fn set_env_for_tests(key: &str, val: &str) {
    // SAFETY: all test env mutation is serialized through env_guard(),
    // and no other thread in this binary mutates the environment.
    unsafe { std::env::set_var(key, val) }
}

/// Test-only env mutation. Callers must hold [`env_guard`].
#[cfg(test)]
pub(crate) fn del_env_for_tests(key: &str) {
    // SAFETY: see set_env_for_tests.
    unsafe { std::env::remove_var(key) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    struct TestSeg {
        name: CString,
        ptr: *mut libc::c_void,
        len: usize,
    }
    impl Drop for TestSeg {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr, self.len);
                libc::shm_unlink(self.name.as_ptr());
            }
        }
    }

    fn make_seg(name: &str, w: u32, h: u32, present_no: u64, time_ns: u64) -> TestSeg {
        make_seg_with(name, w, h, present_no, time_ns, |_| {})
    }

    fn make_seg_with(
        name: &str,
        w: u32,
        h: u32,
        present_no: u64,
        time_ns: u64,
        tweak: impl FnOnce(&mut XdShmHeader),
    ) -> TestSeg {
        let cname = CString::new(name).unwrap();
        unsafe { libc::shm_unlink(cname.as_ptr()) };
        let len = HEADER_SIZE + w as usize * h as usize * 4;
        let fd = unsafe {
            libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            )
        };
        assert!(fd >= 0, "shm_open failed");
        assert_eq!(unsafe { libc::ftruncate(fd, len as libc::off_t) }, 0);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        unsafe { libc::close(fd) };
        // Writer protocol: seq odd -> header+pixels -> seq even.
        let mut hdr = XdShmHeader {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            seq: 1,
            w,
            h,
            format: FORMAT_D32F,
            eye_layout: EYE_SIDEBYSIDE,
            present_no,
            time_mono_ns: time_ns,
            min_depth: 0.0,
            max_depth: 0.5,
            reserved: [0, 0],
        };
        tweak(&mut hdr);
        let pixels: Vec<f32> = (0..w as usize * h as usize)
            .map(|i| i as f32 / 1000.0)
            .collect();
        publish(ptr, &mut hdr, &pixels);
        TestSeg {
            name: cname,
            ptr,
            len,
        }
    }

    /// Mirror of the producer publish sequence (odd -> copy -> even).
    fn publish(ptr: *mut libc::c_void, hdr: &mut XdShmHeader, pixels: &[f32]) {
        assert_eq!(pixels.len(), hdr.w as usize * hdr.h as usize);
        hdr.seq = 1;
        unsafe {
            (ptr as *mut XdShmHeader).write_unaligned(*hdr);
            if !pixels.is_empty() {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr(),
                    (ptr as *mut u8).add(HEADER_SIZE) as *mut f32,
                    pixels.len(),
                );
            }
            // publish
            (ptr as *mut u32).add(2).write_volatile(2);
        }
    }

    #[test]
    fn header_is_64_bytes() {
        assert_eq!(std::mem::size_of::<XdShmHeader>(), 64);
    }

    #[test]
    fn fixture_roundtrip() {
        let name = format!("/xrizer-depth-test-{}-1", std::process::id());
        let now = now_mono_ns();
        assert!(now > 0);
        let _seg = make_seg(&name, 8, 2, 42, now);
        let feed = DepthFeed::open_for_test(&name);
        let f = feed.latest_for(now + 1_000_000, 100).expect("should hit");
        assert_eq!((f.w, f.h, f.present_no), (8, 2, 42));
        assert_eq!(f.pixels.len(), 16);
        assert!((f.pixels[0] - 0.0).abs() < f32::EPSILON);
        assert!((f.pixels[15] - 0.015).abs() < 1e-6);
    }

    #[test]
    fn stale_and_future_rejected() {
        let name = format!("/xrizer-depth-test-{}-2", std::process::id());
        let now = now_mono_ns();
        let _seg = make_seg(&name, 4, 2, 7, now.saturating_sub(500_000_000));
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.latest_for(now, 100).is_none(), "stale must miss");
        // Future stamp must also miss (present newer than color frame).
        let name2 = format!("/xrizer-depth-test-{}-3", std::process::id());
        let _seg2 = make_seg(&name2, 4, 2, 8, now + 50_000_000);
        let feed2 = DepthFeed::open_for_test(&name2);
        assert!(feed2.latest_for(now, 100).is_none(), "future must miss");
    }

    #[test]
    fn bad_magic_rejected() {
        let name = format!("/xrizer-depth-test-{}-4", std::process::id());
        let now = now_mono_ns();
        let seg = make_seg(&name, 4, 2, 1, now);
        unsafe {
            (seg.ptr as *mut u32).write_volatile(0xdead_beef);
            (seg.ptr as *mut u32).add(2).write_volatile(2);
        }
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.latest_for(now + 1_000_000, 100).is_none());
    }

    #[test]
    fn absent_returns_none() {
        let feed = DepthFeed::open_for_test("/xrizer-depth-test-definitely-absent");
        assert!(feed.latest_for(now_mono_ns(), 100).is_none());
    }

    #[test]
    fn bad_version_format_layout_rejected() {
        let now = now_mono_ns();
        let pid = std::process::id();
        let cases: [(u32, fn(&mut XdShmHeader)); 3] = [
            (5, |h: &mut XdShmHeader| h.version = 2),
            (6, |h: &mut XdShmHeader| h.format = 96),
            (7, |h: &mut XdShmHeader| h.eye_layout = 0),
        ];
        for (suffix, tweak) in cases {
            let name = format!("/xrizer-depth-test-{pid}-{suffix}");
            let _seg = make_seg_with(&name, 4, 2, 1, now, tweak);
            let feed = DepthFeed::open_for_test(&name);
            assert!(
                feed.latest_for(now + 1_000_000, 100).is_none(),
                "segment {suffix} must miss"
            );
        }
    }

    #[test]
    fn bad_dims_rejected() {
        let now = now_mono_ns();
        let pid = std::process::id();
        // Odd width and zero height both fail the plausibility gate.
        for (suffix, w, h) in [(8u32, 5u32, 2u32), (9, 0, 2)] {
            let name = format!("/xrizer-depth-test-{pid}-{suffix}");
            let _seg = make_seg(&name, w, h, 1, now);
            let feed = DepthFeed::open_for_test(&name);
            assert!(
                feed.latest_for(now + 1_000_000, 100).is_none(),
                "dims {w}x{h} must miss"
            );
        }
    }

    #[test]
    fn header_plausible_unit() {
        let base = XdShmHeader {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            seq: 2,
            w: 8,
            h: 2,
            format: FORMAT_D32F,
            eye_layout: EYE_SIDEBYSIDE,
            present_no: 0,
            time_mono_ns: 0,
            min_depth: 0.0,
            max_depth: 1.0,
            reserved: [0, 0],
        };
        let exact = HEADER_SIZE + 8 * 2 * 4;
        assert!(header_plausible(&base, exact));
        assert!(!header_plausible(&base, exact + 4), "size mismatch");
        assert!(!header_plausible(&XdShmHeader { w: 7, ..base }, exact));
        assert!(!header_plausible(&XdShmHeader { w: 0, ..base }, exact));
        assert!(!header_plausible(
            &XdShmHeader { h: 0, ..base },
            HEADER_SIZE
        ));
        assert!(!header_plausible(
            &XdShmHeader { w: 16385, ..base },
            HEADER_SIZE + 16385 * 2 * 4
        ));
    }

    #[test]
    fn stuck_odd_seqlock_returns_none() {
        let name = format!("/xrizer-depth-test-{}-10", std::process::id());
        let now = now_mono_ns();
        let seg = make_seg(&name, 4, 2, 1, now);
        // Writer torn mid-publish: seq stuck odd. Must miss (bounded), not hang.
        unsafe {
            (seg.ptr as *mut u32).add(2).write_volatile(3);
        }
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.latest_for(now + 1_000_000, 100).is_none());
    }

    #[test]
    fn republish_newest_wins() {
        let name = format!("/xrizer-depth-test-{}-11", std::process::id());
        let now = now_mono_ns();
        let seg = make_seg(&name, 4, 2, 1, now);
        let feed = DepthFeed::open_for_test(&name);
        let first = feed.latest_for(now + 1_000_000, 100).expect("first hit");
        assert_eq!(first.present_no, 1);
        // Second publish overwrites the slot; the reader must see it.
        let mut hdr = XdShmHeader {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            seq: 1,
            w: 4,
            h: 2,
            format: FORMAT_D32F,
            eye_layout: EYE_SIDEBYSIDE,
            present_no: 2,
            time_mono_ns: now,
            min_depth: 0.0,
            max_depth: 0.25,
            reserved: [0, 0],
        };
        let pixels = vec![0.5f32; 8];
        publish(seg.ptr, &mut hdr, &pixels);
        let second = feed.latest_for(now + 1_000_000, 100).expect("second hit");
        assert_eq!(second.present_no, 2);
        assert!(
            second
                .pixels
                .iter()
                .all(|&p| (p - 0.5).abs() < f32::EPSILON)
        );
    }

    /// v2 segment builder. `with_pixels` appends the w*h float plane at +128
    /// (PIXELS mode); FD mode segments are exactly 128 bytes.
    fn make_seg_v2(
        name: &str,
        w: u32,
        h: u32,
        generation: u64,
        mode: u32,
        with_pixels: bool,
        tweak: impl FnOnce(&mut XdShmHeaderV2),
    ) -> TestSeg {
        let cname = CString::new(name).unwrap();
        unsafe { libc::shm_unlink(cname.as_ptr()) };
        let len = HEADER_SIZE_V2
            + if with_pixels {
                w as usize * h as usize * 4
            } else {
                0
            };
        let fd = unsafe {
            libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            )
        };
        assert!(fd >= 0, "shm_open failed");
        assert_eq!(unsafe { libc::ftruncate(fd, len as libc::off_t) }, 0);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        unsafe { libc::close(fd) };
        let mut hdr = XdShmHeaderV2 {
            magic: SHM_MAGIC,
            version: SHM_VERSION_V2,
            seq: 1,
            w,
            h,
            format: FORMAT_D32F,
            eye_layout: EYE_SIDEBYSIDE,
            present_no: generation,
            time_mono_ns: now_mono_ns(),
            min_depth: 0.0,
            max_depth: 0.5,
            _pad56: [0, 0],
            mode,
            setup_epoch: 7,
            consumed_gen: 0,
            mem_fds: [11, 12, 13],
            sem_fds: [21, 22, 23],
            near_z: 0.5,
            far_z: 2000.0,
            reserved: [0, 0, 0, 0],
        };
        tweak(&mut hdr);
        unsafe {
            (ptr as *mut XdShmHeaderV2).write_unaligned(hdr);
            if with_pixels {
                let px = (ptr as *mut u8).add(HEADER_SIZE_V2) as *mut f32;
                for i in 0..(w as usize * h as usize) {
                    px.add(i).write((i as f32) / 100.0);
                }
            }
            (ptr as *mut u32).add(2).write_volatile(2);
        }
        TestSeg {
            name: cname,
            ptr,
            len,
        }
    }

    #[test]
    fn v2_pixels_hit_at_plus_128() {
        let name = format!("/xrizer-depth-test-{}-20", std::process::id());
        let now = now_mono_ns();
        let _seg = make_seg_v2(&name, 8, 2, 3, MODE_PIXELS, true, |_| {});
        let feed = DepthFeed::open_for_test(&name);
        let f = feed
            .latest_for(now + 1_000_000, 100)
            .expect("v2 pixels hit");
        assert_eq!((f.w, f.h, f.present_no), (8, 2, 3));
        assert_eq!(f.pixels.len(), 16);
        assert_eq!(f.vk_format, FORMAT_D32F);
        assert!((f.near_z - 0.5).abs() < f32::EPSILON && (f.far_z - 2000.0).abs() < 1.0);
    }

    #[test]
    fn v2_fd_mode_pixels_path_misses() {
        let name = format!("/xrizer-depth-test-{}-21", std::process::id());
        let _seg = make_seg_v2(&name, 8, 2, 3, MODE_FD, false, |_| {});
        let feed = DepthFeed::open_for_test(&name);
        assert!(
            feed.latest_for(now_mono_ns() + 1_000_000, 100).is_none(),
            "FD mode has no pixel plane"
        );
    }

    #[test]
    fn unknown_version_misses() {
        let name = format!("/xrizer-depth-test-{}-22", std::process::id());
        let now = now_mono_ns();
        let _seg = make_seg_with(&name, 4, 2, 1, now, |h| h.version = 3);
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.latest_for(now + 1_000_000, 100).is_none());
        let ctl = feed.read_control();
        assert!(ctl.is_none(), "unknown version has no control plane");
    }

    #[test]
    fn read_control_v1_and_v2() {
        let pid = std::process::id();
        let now = now_mono_ns();
        let name1 = format!("/xrizer-depth-test-{pid}-23");
        let _seg1 = make_seg(&name1, 6, 2, 9, now);
        let ctl1 = DepthFeed::open_for_test(&name1)
            .read_control()
            .expect("v1 control");
        assert_eq!(
            (ctl1.version, ctl1.mode, ctl1.setup_epoch),
            (1, MODE_PIXELS, 0)
        );
        assert_eq!((ctl1.w, ctl1.h, ctl1.generation), (6, 2, 9));
        assert_eq!(ctl1.mem_fds, [-1; 3]);

        let name2 = format!("/xrizer-depth-test-{pid}-24");
        let _seg2 = make_seg_v2(&name2, 8, 2, 42, MODE_FD, false, |_| {});
        let ctl2 = DepthFeed::open_for_test(&name2)
            .read_control()
            .expect("v2 control");
        assert_eq!((ctl2.version, ctl2.mode, ctl2.setup_epoch), (2, MODE_FD, 7));
        assert_eq!((ctl2.w, ctl2.h, ctl2.generation), (8, 2, 42));
        assert_eq!(ctl2.mem_fds, [11, 12, 13]);
        assert_eq!(ctl2.sem_fds, [21, 22, 23]);
        assert!((ctl2.near_z - 0.5).abs() < f32::EPSILON);
        assert!((ctl2.far_z - 2000.0).abs() < 1.0);
    }

    #[test]
    fn store_consumed_roundtrip() {
        let name = format!("/xrizer-depth-test-{}-25", std::process::id());
        let _seg = make_seg_v2(&name, 8, 2, 42, MODE_FD, false, |_| {});
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.store_consumed(42));
        let ctl = feed.read_control().expect("control readable");
        assert_eq!(ctl.consumed_gen, 42);
    }

    #[test]
    fn resolve_near_far_prefers_header() {
        let _lock = env_guard();
        let (near, far) = resolve_near_far(0.5, 2000.0);
        assert!((near - 0.5).abs() < f32::EPSILON && (far - 2000.0).abs() < 1.0);
        let (a_near, a_far) = resolve_near_far(0.0, 0.0);
        let (b_near, b_far) = near_far();
        assert!((a_near - b_near).abs() < f32::EPSILON && (a_far - b_far).abs() < 1.0);
        del_env_for_tests("XRIZER_DEPTH_NEAR");
        del_env_for_tests("XRIZER_DEPTH_FAR");
        assert_eq!(resolve_near_far(0.0, 100.0), (1.0, 30000.0));
    }

    #[test]
    fn env_parsing() {
        let _lock = env_guard();
        let save = |key: &str| std::env::var(key).ok();
        let orig_master = save("XRIZER_DEPTH_SHM");
        let orig_name = save("XP_DEPTH_SHM_NAME");
        let orig_age = save("XRIZER_DEPTH_MAX_AGE_MS");
        let orig_near = save("XRIZER_DEPTH_NEAR");
        let orig_far = save("XRIZER_DEPTH_FAR");
        let restore = |key: &str, val: Option<String>| match val {
            Some(v) => set_env_for_tests(key, &v),
            None => del_env_for_tests(key),
        };

        del_env_for_tests("XRIZER_DEPTH_SHM");
        assert!(!depth_master_enabled());
        set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        assert!(depth_master_enabled());
        set_env_for_tests("XRIZER_DEPTH_SHM", "0");
        assert!(!depth_master_enabled());
        set_env_for_tests("XRIZER_DEPTH_SHM", "yes");
        assert!(!depth_master_enabled());

        del_env_for_tests("XP_DEPTH_SHM_NAME");
        assert_eq!(shm_name(), DEFAULT_SHM_NAME);
        set_env_for_tests("XP_DEPTH_SHM_NAME", "/custom-depth");
        assert_eq!(shm_name(), "/custom-depth");

        del_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS");
        assert_eq!(max_age_ms(), 100);
        set_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS", "250");
        assert_eq!(max_age_ms(), 250);
        set_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS", "abc");
        assert_eq!(max_age_ms(), 100);

        del_env_for_tests("XRIZER_DEPTH_NEAR");
        del_env_for_tests("XRIZER_DEPTH_FAR");
        let (near, far) = near_far();
        assert!((near - 1.0).abs() < f32::EPSILON && (far - 30000.0).abs() < 1.0);
        set_env_for_tests("XRIZER_DEPTH_NEAR", "0.5");
        set_env_for_tests("XRIZER_DEPTH_FAR", "1000.0");
        let (near, far) = near_far();
        assert!((near - 0.5).abs() < f32::EPSILON && (far - 1000.0).abs() < f32::EPSILON);
        set_env_for_tests("XRIZER_DEPTH_NEAR", "bogus");
        let (near, far) = near_far();
        assert!((near - 1.0).abs() < f32::EPSILON && (far - 1000.0).abs() < f32::EPSILON);

        restore("XRIZER_DEPTH_SHM", orig_master);
        restore("XP_DEPTH_SHM_NAME", orig_name);
        restore("XRIZER_DEPTH_MAX_AGE_MS", orig_age);
        restore("XRIZER_DEPTH_NEAR", orig_near);
        restore("XRIZER_DEPTH_FAR", orig_far);
    }
}
