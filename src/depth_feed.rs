//! SHM depth handoff reader (reprojex layer -> xrizer), FD mode only.
//!
//! Transport is a POSIX SHM 128-byte control header (`/xplane-depth-v1`,
//! fixed — no env override; a rendezvous point must not be configurable).
//! No pixels travel here: depth lives in layer-owned GPU mirror images
//! shared by FD (same process: FD numbers are used directly).
//!
//! Layout is little-endian, all offsets fixed (byte-identical with
//! `reprojex/reprojex_shm.h`; the `const _` asserts below pin it):
//!
//! ```text
//! +0  u32 magic       0x58504450 ('XPDP')
//! +4  u32 version     2 (v1 deleted — shim only writes v2)
//! +8  u32 seq         seqlock: odd = writer inside
//! +12 u32 w, +16 u32 h
//! +20 i32 format      126 = D32F only
//! +24 u32 eye_layout  1 = side-by-side (left half eye0, right half eye1)
//! +32 u64 present_no  latest signaled generation (0 = never published)
//! +40 u64 time_mono_ns (CLOCK_MONOTONIC at copy completion)
//! +48 f32 min_depth, +52 f32 max_depth (mapping range, never gated)
//! +56 u32 pad[2]
//! +64 u32 mode        1 = FD (anything else → not viable, plain views)
//! +68 u32 setup_epoch (0 = never published; reimport on change)
//! +72 u64 consumed_gen (written by xrizer: highest fully-copied gen)
//! +80 i32 mem_fds[3], +92 i32 sem_fds[3] (pre-publish sentinel -1)
//! +104 f32 near_z, +108 f32 far_z (header-only; invalid → error + plain)
//! +112 u32 alloc_bytes (import size must be <= this; 0 = unknown)
//! +116 u32 reserved[3]
//! ```
//!
//! Reader discipline: seqlock retry (odd/even, bounded), `fstat` size must
//! be exactly 128, magic/version checked. Absent/stale/mismatched SHM
//! always yields `None` so callers fall back to plain (color-only) views.
//! Never blocks the app thread: bounded retries, no waits.
//!
//! No new crates: `libc` + `log` only.

use log::{debug, info, warn};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const SHM_MAGIC: u32 = 0x5850_4450; // 'XPDP'
pub const SHM_VERSION_V2: u32 = 2;
pub const EYE_SIDEBYSIDE: u32 = 1;
pub const MODE_FD: u32 = 1;
/// Vulkan `VK_FORMAT_D32_SFLOAT`. Only format accepted in prod.
pub const FORMAT_D32F: i32 = 126;
pub const HEADER_SIZE_V2: usize = 128;
/// Byte offset of `consumed_gen`: written by xrizer, 8-aligned.
pub const CONSUMED_GEN_OFF: usize = 72;
/// Fixed rendezvous. No env override.
pub const DEFAULT_SHM_NAME: &str = "/xplane-depth-v1";
/// Default staleness bound (ms) when `XRIZER_DEPTH_MAX_AGE_MS` is unset.
pub const DEFAULT_MAX_AGE_MS: u64 = 100;

/// Mirror of the C `rx_shm_header`. Must stay 128 bytes with the exact
/// offsets below (`mode` at +64, `consumed_gen` at +72, `alloc_bytes`
/// at +112). Keep in sync with `reprojex/reprojex_shm.h`.
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
    pub present_no: u64,    // +32
    pub time_mono_ns: u64,  // +40
    pub min_depth: f32,     // +48
    pub max_depth: f32,     // +52
    pub _pad56: [u32; 2],   // +56,+60
    pub mode: u32,          // +64
    pub setup_epoch: u32,   // +68
    pub consumed_gen: u64,  // +72 xrizer -> shim
    pub mem_fds: [i32; 3],  // +80
    pub sem_fds: [i32; 3],  // +92
    pub near_z: f32,        // +104
    pub far_z: f32,         // +108
    pub alloc_bytes: u32,   // +112
    pub reserved: [u32; 3], // +116,+120,+124
}

// Pinned at build time, not just under test: a drifted offset is a
// cross-process ABI break, not a test failure.
const _: () = {
    assert!(std::mem::size_of::<XdShmHeaderV2>() == HEADER_SIZE_V2);
    assert!(std::mem::offset_of!(XdShmHeaderV2, magic) == 0);
    assert!(std::mem::offset_of!(XdShmHeaderV2, version) == 4);
    assert!(std::mem::offset_of!(XdShmHeaderV2, seq) == 8);
    assert!(std::mem::offset_of!(XdShmHeaderV2, w) == 12);
    assert!(std::mem::offset_of!(XdShmHeaderV2, h) == 16);
    assert!(std::mem::offset_of!(XdShmHeaderV2, format) == 20);
    assert!(std::mem::offset_of!(XdShmHeaderV2, eye_layout) == 24);
    assert!(std::mem::offset_of!(XdShmHeaderV2, present_no) == 32);
    assert!(std::mem::offset_of!(XdShmHeaderV2, time_mono_ns) == 40);
    assert!(std::mem::offset_of!(XdShmHeaderV2, min_depth) == 48);
    assert!(std::mem::offset_of!(XdShmHeaderV2, max_depth) == 52);
    assert!(std::mem::offset_of!(XdShmHeaderV2, mode) == 64);
    assert!(std::mem::offset_of!(XdShmHeaderV2, setup_epoch) == 68);
    assert!(std::mem::offset_of!(XdShmHeaderV2, consumed_gen) == CONSUMED_GEN_OFF);
    assert!(std::mem::offset_of!(XdShmHeaderV2, mem_fds) == 80);
    assert!(std::mem::offset_of!(XdShmHeaderV2, sem_fds) == 92);
    assert!(std::mem::offset_of!(XdShmHeaderV2, near_z) == 104);
    assert!(std::mem::offset_of!(XdShmHeaderV2, far_z) == 108);
    assert!(std::mem::offset_of!(XdShmHeaderV2, alloc_bytes) == 112);
    assert!(std::mem::offset_of!(XdShmHeaderV2, reserved) == 116);
};

/// Snapshot of the v2 control header for the FD path.
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
    pub min_depth: f32,
    pub max_depth: f32,
    pub mem_fds: [i32; 3],
    pub sem_fds: [i32; 3],
    pub near_z: f32,
    pub far_z: f32,
    /// Highest fully-copied generation, as last observed.
    pub consumed_gen: u64,
    /// Observed segment length; viable only when exactly 128.
    pub seg_len: usize,
    /// Driver-reported mirror allocation size (+112). Import size must be
    /// `<= alloc_bytes` when non-zero; 0 = unknown (tolerate).
    pub alloc_bytes: u32,
}

impl DepthControl {
    /// Single-eye width of the side-by-side mirror.
    pub fn eye_w(&self) -> u32 {
        self.w / 2
    }

    /// All six FDs must be real (stdin/stdout/stderr are never depth).
    /// The shim's pre-publish sentinel is -1.
    pub fn fds_valid(&self) -> bool {
        self.mem_fds.iter().all(|&f| f > 2) && self.sem_fds.iter().all(|&f| f > 2)
    }

    pub fn dims_plausible(&self) -> bool {
        self.w != 0 && self.h != 0 && self.w <= 16384 && self.h <= 16384 && self.w.is_multiple_of(2)
    }

    /// Full viability gate: v2 FD, published epoch+generation, D32F SBS,
    /// plausible dims, exact 128-byte segment, all FDs real.
    /// `min/max` are mapping range (never gated); `near/far` validity is
    /// checked separately via [`validate_near_far`].
    pub fn is_viable(&self) -> bool {
        self.version == SHM_VERSION_V2
            && self.mode == MODE_FD
            && self.setup_epoch != 0
            && self.generation != 0
            && self.format == FORMAT_D32F
            && self.eye_layout == EYE_SIDEBYSIDE
            && self.seg_len == HEADER_SIZE_V2
            && self.dims_plausible()
            && self.fds_valid()
    }

    /// Staleness against the header timestamp. Future stamps and zero
    /// stamps are stale (clock skew or never published).
    pub fn is_stale(&self, now_mono_ns: u64, max_age_ms: u64) -> bool {
        if self.time_ns == 0 || self.time_ns > now_mono_ns {
            return true;
        }
        now_mono_ns.saturating_sub(self.time_ns) > max_age_ms.saturating_mul(1_000_000)
    }
}

/// Reader handle. Stateless across frames (open/mmap per read) so producer
/// `munmap/shm_unlink/recreate` cycles on geometry changes are picked up
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

// SAFETY: the mapping is only read through volatile/checked seqlock copies
// on the calling thread; the `MappedSeg` itself is never shared.
unsafe impl Send for MappedSeg {}

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
    // FD mode needs exactly the 128-byte header: anything else belongs to
    // a dead run (unlinked, mid-recreate) or a foreign writer.
    if map_len != HEADER_SIZE_V2 {
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
    /// `None` when the `XRIZER_DEPTH_SHM` kill-switch is set (`=0` etc).
    /// Default ON: depth arms whenever the runtime supports it.
    pub fn open() -> Option<Self> {
        if !depth_master_enabled() {
            return None;
        }
        if !FEED_OPEN_LOGGED.swap(true, Ordering::Relaxed) {
            info!("depth feed: SHM path {} (master on)", DEFAULT_SHM_NAME);
        }
        Some(Self {
            name: CString::new(DEFAULT_SHM_NAME).expect("SHM name contains NUL"),
        })
    }

    /// Test seam: custom segment name (prod path is always fixed).
    #[cfg(test)]
    fn open_for_test(name: &str) -> Self {
        Self {
            name: CString::new(name).expect("SHM name contains NUL"),
        }
    }

    /// Seqlock-read of the v2 control header. No pixel copy (FD mode has
    /// no pixel plane). Returns `None` on absent/wrong-size/torn/foreign
    /// data so callers fall back to plain views.
    pub fn read_control(&self) -> Option<DepthControl> {
        let map = map_segment(&self.name, false)?;
        let (ptr, map_len) = (map.ptr, map.len);
        debug_assert_eq!(map_len, HEADER_SIZE_V2);
        for _ in 0..4 {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            if unsafe { read_u32(ptr, 0) } != SHM_MAGIC {
                return None;
            }
            if unsafe { read_u32(ptr, 4) } != SHM_VERSION_V2 {
                return None;
            }
            // Copy under seq1; the closing seq check covers torn reads.
            let h = unsafe { read_header_v2(ptr) };
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            // Trust only the copied snapshot from here on — except the ack:
            // consumed_gen is written concurrently (by us via release-store;
            // preserved — never zeroed — by the producer across recreates),
            // so reload it with an acquire-load pairing our store.
            // mmap is page-aligned and CONSUMED_GEN_OFF is 8-aligned,
            // so this is always aligned.
            if h.magic != SHM_MAGIC || h.version != SHM_VERSION_V2 {
                return None;
            }
            let consumed_gen = unsafe {
                (ptr as *const u8)
                    .add(CONSUMED_GEN_OFF)
                    .cast::<AtomicU64>()
                    .as_ref()
                    .map(|a| a.load(Ordering::Acquire))
                    .unwrap_or(h.consumed_gen)
            };
            return Some(DepthControl {
                version: h.version,
                mode: h.mode,
                setup_epoch: h.setup_epoch,
                w: h.w,
                h: h.h,
                format: h.format,
                eye_layout: h.eye_layout,
                generation: h.present_no,
                time_ns: h.time_mono_ns,
                min_depth: h.min_depth,
                max_depth: h.max_depth,
                mem_fds: h.mem_fds,
                sem_fds: h.sem_fds,
                near_z: h.near_z,
                far_z: h.far_z,
                consumed_gen,
                seg_len: map_len,
                alloc_bytes: h.alloc_bytes,
            });
        }
        None
    }

    /// Record the highest fully-copied generation. Release store pairs
    /// with the producer's acquire-load; a lost update only skips a slot
    /// (benign). Always stores (bounded even-seq spin first): a missed
    /// ack stalls the producer ring. Call ONLY after both eyes copied.
    pub fn store_consumed(&self, generation: u64) -> bool {
        let Some(map) = map_segment(&self.name, true) else {
            return false;
        };
        if map.len != HEADER_SIZE_V2 {
            return false;
        }
        for _ in 0..100 {
            if unsafe { read_seq(map.ptr) } & 1 == 0 {
                break;
            }
            std::hint::spin_loop();
        }
        // Never store into a torn publish: the writer preserves the ack
        // across recreates, so a missed store is a benign skipped tick.
        if unsafe { read_seq(map.ptr) } & 1 != 0 {
            return false;
        }
        debug_assert_eq!(map.len, HEADER_SIZE_V2);
        // Release assert: the AtomicU64 cast below faults on misalignment
        // on strict architectures; page mappings are aligned in practice,
        // but never skip this check in release.
        assert_eq!(
            (map.ptr as usize + CONSUMED_GEN_OFF) % std::mem::align_of::<AtomicU64>(),
            0
        );
        unsafe {
            if let Some(a) = (map.ptr as *mut u8)
                .add(CONSUMED_GEN_OFF)
                .cast::<AtomicU64>()
                .as_ref()
            {
                a.store(generation, Ordering::Release);
            }
        }
        true
    }
}

unsafe fn read_seq(base: *mut libc::c_void) -> u32 {
    unsafe { (base as *const u8).add(8).cast::<u32>().read_volatile() }
}

unsafe fn read_u32(base: *mut libc::c_void, off: usize) -> u32 {
    // Seqlock-protected header words: the writer mutates them concurrently,
    // so use volatile reads (plain .read() would be a data race).
    unsafe { (base as *const u8).add(off).cast::<u32>().read_volatile() }
}

unsafe fn read_header_v2(base: *mut libc::c_void) -> XdShmHeaderV2 {
    // Same: torn reads are detected via the seqlock retry, but the read
    // itself must be volatile to be sound under concurrent mutation.
    unsafe { (base as *const XdShmHeaderV2).read_volatile() }
}

/// Header near/far win — no xrizer-side defaults. Pure predicate: invalid
/// (`<= 0`, non-finite, `near >= far`) yields `None` so the caller falls
/// back to plain views; the caller owns logging (error on the submit
/// path, debug on the quiet upgrade path).
pub fn validate_near_far(hdr_near: f32, hdr_far: f32) -> Option<(f32, f32)> {
    if !hdr_near.is_finite() || !hdr_far.is_finite() {
        return None;
    }
    if hdr_near <= 0.0 || hdr_far <= 0.0 {
        return None;
    }
    if hdr_near >= hdr_far {
        return None;
    }
    Some((hdr_near, hdr_far))
}

/// Header mapping range, sanitized for the runtime (Monado rejects frames
/// on non-finite min/max). Out-of-range or inverted ranges fall back to
/// the identity (0.0, 1.0); the range is never used to gate depth.
pub fn sanitize_min_max(hdr_min: f32, hdr_max: f32) -> (f32, f32) {
    if hdr_min.is_finite()
        && hdr_max.is_finite()
        && hdr_min >= 0.0
        && hdr_max <= 1.0
        && hdr_min <= hdr_max
    {
        (hdr_min, hdr_max)
    } else {
        debug!("depth min/max ({hdr_min}, {hdr_max}) out of range; using 0.0/1.0");
        (0.0, 1.0)
    }
}

static WARNED_MASTER: AtomicBool = AtomicBool::new(false);
static WARNED_MAX_AGE: AtomicBool = AtomicBool::new(false);
static FEED_OPEN_LOGGED: AtomicBool = AtomicBool::new(false);

fn warn_once_flag(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        warn!("{msg}");
    }
}

/// Master switch, default ON. Explicit `0/false/no/off` disables;
/// anything unparseable warns once and stays ON (fail-open toward depth).
pub fn depth_master_enabled() -> bool {
    match std::env::var("XRIZER_DEPTH_SHM") {
        Err(_) => true,
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            match t.as_str() {
                "0" | "false" | "no" | "off" => false,
                "1" | "true" | "yes" | "on" => true,
                _ => {
                    warn_once_flag(
                        &WARNED_MASTER,
                        &format!("XRIZER_DEPTH_SHM={v:?} unparseable; defaulting ON"),
                    );
                    true
                }
            }
        }
    }
}

/// Staleness bound in ms. Garbage/zero/negative/huge warns once and
/// falls back to 100.
pub fn max_age_ms() -> u64 {
    match std::env::var("XRIZER_DEPTH_MAX_AGE_MS") {
        Err(_) => DEFAULT_MAX_AGE_MS,
        Ok(v) => match v.trim().parse::<i64>() {
            Ok(n) if (1..=60_000).contains(&n) => n as u64,
            _ => {
                warn_once_flag(
                    &WARNED_MAX_AGE,
                    &format!("XRIZER_DEPTH_MAX_AGE_MS={v:?} invalid; defaulting 100"),
                );
                DEFAULT_MAX_AGE_MS
            }
        },
    }
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

/// Serializes tests that mutate process-global env vars.
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

    fn make_seg_v2(
        name: &str,
        w: u32,
        h: u32,
        generation: u64,
        mode: u32,
        tweak: impl FnOnce(&mut XdShmHeaderV2),
    ) -> TestSeg {
        let cname = CString::new(name).unwrap();
        unsafe { libc::shm_unlink(cname.as_ptr()) };
        let len = HEADER_SIZE_V2;
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
            max_depth: 1.0,
            _pad56: [0, 0],
            mode,
            setup_epoch: 7,
            consumed_gen: 0,
            mem_fds: [11, 12, 13],
            sem_fds: [21, 22, 23],
            near_z: 0.01,
            far_z: 200.0,
            alloc_bytes: 8 * 2 * 4,
            reserved: [0, 0, 0],
        };
        tweak(&mut hdr);
        unsafe {
            (ptr as *mut XdShmHeaderV2).write_unaligned(hdr);
            (ptr as *mut u32).add(2).write_volatile(2);
        }
        TestSeg {
            name: cname,
            ptr,
            len,
        }
    }

    fn make_raw_seg(name: &str, len: usize, fill: impl FnOnce(*mut libc::c_void)) -> TestSeg {
        let cname = CString::new(name).unwrap();
        unsafe { libc::shm_unlink(cname.as_ptr()) };
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
        fill(ptr);
        TestSeg {
            name: cname,
            ptr,
            len,
        }
    }

    #[test]
    fn header_layout_matches_shm_contract() {
        // Byte-identical with reprojex/reprojex_shm.h (v2, 128B).
        assert_eq!(std::mem::size_of::<XdShmHeaderV2>(), 128);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, magic), 0);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, version), 4);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, seq), 8);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, w), 12);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, h), 16);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, format), 20);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, eye_layout), 24);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, present_no), 32);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, time_mono_ns), 40);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, min_depth), 48);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, max_depth), 52);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, mode), 64);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, setup_epoch), 68);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, consumed_gen), 72);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, mem_fds), 80);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, sem_fds), 92);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, near_z), 104);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, far_z), 108);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, alloc_bytes), 112);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV2, reserved), 116);
    }

    #[test]
    fn roundtrip_viable() {
        let name = format!("/xrizer-depth-test-{}-1", std::process::id());
        let _seg = make_seg_v2(&name, 8, 2, 42, MODE_FD, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("should hit");
        assert_eq!((ctl.w, ctl.h, ctl.generation), (8, 2, 42));
        assert_eq!(ctl.mem_fds, [11, 12, 13]);
        assert_eq!(ctl.sem_fds, [21, 22, 23]);
        assert_eq!(ctl.alloc_bytes, 8 * 2 * 4);
        assert!((ctl.near_z - 0.01).abs() < f32::EPSILON);
        assert!((ctl.far_z - 200.0).abs() < f32::EPSILON);
        assert!((ctl.min_depth - 0.0).abs() < f32::EPSILON);
        assert!((ctl.max_depth - 1.0).abs() < f32::EPSILON);
        assert!(ctl.is_viable());
        assert_eq!(ctl.eye_w(), 4);
        assert!(!ctl.is_stale(now_mono_ns(), 100));
    }

    #[test]
    fn v1_rejected() {
        let pid = std::process::id();
        // 64-byte v1-shaped segment: wrong size entirely.
        let name = format!("/xrizer-depth-test-{pid}-v1a");
        let _seg = make_raw_seg(&name, 64, |ptr| unsafe {
            let mut hdr = [0u32; 16];
            hdr[0] = SHM_MAGIC;
            hdr[1] = 1; // v1
            hdr[2] = 2; // even seq
            std::ptr::copy_nonoverlapping(hdr.as_ptr(), ptr as *mut u32, 16);
        });
        assert!(
            DepthFeed::open_for_test(&name).read_control().is_none(),
            "v1 size must miss"
        );
        // 128-byte segment stamped version 1: wrong version.
        let name2 = format!("/xrizer-depth-test-{pid}-v1b");
        let _seg2 = make_raw_seg(&name2, HEADER_SIZE_V2, |ptr| unsafe {
            let mut hdr = XdShmHeaderV2 {
                magic: SHM_MAGIC,
                version: 1,
                seq: 2,
                w: 8,
                h: 2,
                format: FORMAT_D32F,
                eye_layout: EYE_SIDEBYSIDE,
                present_no: 1,
                time_mono_ns: now_mono_ns(),
                min_depth: 0.0,
                max_depth: 1.0,
                _pad56: [0, 0],
                mode: MODE_FD,
                setup_epoch: 7,
                consumed_gen: 0,
                mem_fds: [11, 12, 13],
                sem_fds: [21, 22, 23],
                near_z: 0.01,
                far_z: 200.0,
                alloc_bytes: 64,
                reserved: [0, 0, 0],
            };
            hdr.seq = 1;
            (ptr as *mut XdShmHeaderV2).write_unaligned(hdr);
            (ptr as *mut u32).add(2).write_volatile(2);
        });
        assert!(
            DepthFeed::open_for_test(&name2).read_control().is_none(),
            "version 1 must miss"
        );
    }

    #[test]
    fn bad_magic_and_size_rejected() {
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-mag");
        let seg = make_seg_v2(&name, 8, 2, 1, MODE_FD, |_| {});
        unsafe {
            (seg.ptr as *mut u32).write_volatile(0xdead_beef);
            (seg.ptr as *mut u32).add(2).write_volatile(2);
        }
        assert!(
            DepthFeed::open_for_test(&name).read_control().is_none(),
            "bad magic must miss"
        );
        // Oversized segment: foreign writer, not our 128B header.
        let name2 = format!("/xrizer-depth-test-{pid}-big");
        let _seg2 = make_raw_seg(&name2, HEADER_SIZE_V2 + 64, |ptr| unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0, HEADER_SIZE_V2 + 64);
            (ptr as *mut u32).write_volatile(SHM_MAGIC);
            (ptr as *mut u32).add(1).write_volatile(SHM_VERSION_V2);
            (ptr as *mut u32).add(2).write_volatile(2);
        });
        assert!(
            DepthFeed::open_for_test(&name2).read_control().is_none(),
            "wrong size must miss"
        );
    }

    #[test]
    fn non_fd_mode_not_viable() {
        let pid = std::process::id();
        for (suffix, mode) in [(40, 0), (41, 7)] {
            let name = format!("/xrizer-depth-test-{pid}-{suffix}");
            let _seg = make_seg_v2(&name, 8, 2, 3, mode, |_| {});
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("header still readable");
            assert_eq!(ctl.mode, mode);
            assert!(!ctl.is_viable(), "mode {mode} must not arm depth");
        }
    }

    #[test]
    fn sentinel_and_low_fds_rejected() {
        let pid = std::process::id();
        let cases: [(i32, i32); 5] = [(-1, 21), (0, 21), (1, 21), (2, 21), (11, 2)];
        for (i, (mem0, sem0)) in cases.iter().enumerate() {
            let name = format!("/xrizer-depth-test-{pid}-fd{i}");
            let _seg = make_seg_v2(&name, 8, 2, 3, MODE_FD, |h| {
                h.mem_fds[0] = *mem0;
                h.sem_fds[0] = *sem0;
            });
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("header readable");
            assert!(
                !ctl.fds_valid() && !ctl.is_viable(),
                "fds ({mem0}, {sem0}) must not arm"
            );
        }
    }

    #[test]
    fn pre_publish_and_dims_rejected() {
        let pid = std::process::id();
        // epoch 0 / generation 0: never published.
        let name = format!("/xrizer-depth-test-{pid}-pre");
        let _seg = make_seg_v2(&name, 8, 2, 0, MODE_FD, |h| {
            h.setup_epoch = 0;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "epoch/gen 0 must not arm");
        // Bad dims / format / layout.
        type HeaderTweak = (&'static str, fn(&mut XdShmHeaderV2));
        let tweaks: [HeaderTweak; 6] = [
            ("oddw", |h: &mut XdShmHeaderV2| h.w = 7),
            ("zerow", |h: &mut XdShmHeaderV2| h.w = 0),
            ("huge", |h: &mut XdShmHeaderV2| h.w = 16385),
            ("fmt", |h: &mut XdShmHeaderV2| h.format = 96),
            ("layout", |h: &mut XdShmHeaderV2| h.eye_layout = 0),
            ("gen0", |h: &mut XdShmHeaderV2| h.present_no = 0),
        ];
        for (tag, tweak) in tweaks {
            let name = format!("/xrizer-depth-test-{pid}-{tag}");
            let _seg = make_seg_v2(&name, 8, 2, 3, MODE_FD, tweak);
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("readable");
            assert!(!ctl.is_viable(), "{tag} must not arm");
        }
    }

    #[test]
    fn stale_and_future_are_stale() {
        let pid = std::process::id();
        let now = now_mono_ns();
        let name = format!("/xrizer-depth-test-{pid}-stale");
        let _seg = make_seg_v2(&name, 8, 2, 7, MODE_FD, |h| {
            h.time_mono_ns = now.saturating_sub(500_000_000);
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable());
        assert!(ctl.is_stale(now, 100), "500ms old must be stale");
        let name2 = format!("/xrizer-depth-test-{pid}-future");
        let _seg2 = make_seg_v2(&name2, 8, 2, 8, MODE_FD, |h| {
            h.time_mono_ns = now.saturating_add(50_000_000);
        });
        let ctl2 = DepthFeed::open_for_test(&name2)
            .read_control()
            .expect("readable");
        assert!(ctl2.is_stale(now, 100), "future stamp must be stale");
    }

    #[test]
    fn store_consumed_roundtrip() {
        let name = format!("/xrizer-depth-test-{}-ack", std::process::id());
        let _seg = make_seg_v2(&name, 8, 2, 42, MODE_FD, |_| {});
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.store_consumed(42));
        let ctl = feed.read_control().expect("control readable");
        assert_eq!(ctl.consumed_gen, 42);
    }

    #[test]
    fn stuck_odd_seqlock_returns_none() {
        let name = format!("/xrizer-depth-test-{}-odd", std::process::id());
        let seg = make_seg_v2(&name, 8, 2, 1, MODE_FD, |_| {});
        unsafe {
            (seg.ptr as *mut u32).add(2).write_volatile(3);
        }
        let feed = DepthFeed::open_for_test(&name);
        assert!(feed.read_control().is_none());
    }

    #[test]
    fn absent_returns_none() {
        let feed = DepthFeed::open_for_test("/xrizer-depth-test-definitely-absent");
        assert!(feed.read_control().is_none());
        assert!(!feed.store_consumed(1));
    }

    #[test]
    fn near_far_strict() {
        assert_eq!(validate_near_far(0.01, 200.0), Some((0.01, 200.0)));
        assert_eq!(validate_near_far(0.5, 2000.0), Some((0.5, 2000.0)));
        assert_eq!(validate_near_far(0.0, 200.0), None);
        assert_eq!(validate_near_far(0.01, 0.0), None);
        assert_eq!(validate_near_far(-1.0, 200.0), None);
        assert_eq!(validate_near_far(200.0, 0.01), None);
        assert_eq!(validate_near_far(1.0, 1.0), None);
        assert_eq!(validate_near_far(f32::NAN, 200.0), None);
        assert_eq!(validate_near_far(0.01, f32::INFINITY), None);
    }

    #[test]
    fn min_max_sanitized() {
        assert_eq!(sanitize_min_max(0.0, 1.0), (0.0, 1.0));
        assert_eq!(sanitize_min_max(0.2, 0.8), (0.2, 0.8));
        assert_eq!(sanitize_min_max(f32::NAN, 1.0), (0.0, 1.0));
        assert_eq!(sanitize_min_max(0.0, f32::INFINITY), (0.0, 1.0));
        assert_eq!(sanitize_min_max(-0.1, 1.0), (0.0, 1.0));
        assert_eq!(sanitize_min_max(0.0, 1.5), (0.0, 1.0));
        assert_eq!(sanitize_min_max(0.8, 0.2), (0.0, 1.0));
    }

    #[test]
    fn env_parsing_strict() {
        let _lock = env_guard();
        let save = |key: &str| std::env::var(key).ok();
        let orig_master = save("XRIZER_DEPTH_SHM");
        let orig_age = save("XRIZER_DEPTH_MAX_AGE_MS");
        let restore = |key: &str, val: Option<String>| match val {
            Some(v) => set_env_for_tests(key, &v),
            None => del_env_for_tests(key),
        };

        // Master default ON.
        del_env_for_tests("XRIZER_DEPTH_SHM");
        assert!(depth_master_enabled());
        for v in ["1", "true", "yes", "on", "YES", " True "] {
            set_env_for_tests("XRIZER_DEPTH_SHM", v);
            assert!(depth_master_enabled(), "{v} enables");
        }
        for v in ["0", "false", "no", "off", "OFF", " False "] {
            set_env_for_tests("XRIZER_DEPTH_SHM", v);
            assert!(!depth_master_enabled(), "{v} disables");
        }
        set_env_for_tests("XRIZER_DEPTH_SHM", "bogus");
        assert!(depth_master_enabled(), "garbage defaults ON");

        del_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS");
        assert_eq!(max_age_ms(), DEFAULT_MAX_AGE_MS);
        set_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS", "250");
        assert_eq!(max_age_ms(), 250);
        for v in ["abc", "0", "-5", "", "99999999"] {
            set_env_for_tests("XRIZER_DEPTH_MAX_AGE_MS", v);
            assert_eq!(max_age_ms(), DEFAULT_MAX_AGE_MS, "{v:?} defaults");
        }

        restore("XRIZER_DEPTH_SHM", orig_master);
        restore("XRIZER_DEPTH_MAX_AGE_MS", orig_age);
    }
}
