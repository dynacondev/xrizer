//! SHM depth-feed reader: xrizer side of the reprojex v3 handoff.
//!
//! Wire contract lives in `reprojex/src/shm.h` (`/xplane-depth-v3`, 96
//! bytes) with reader ordering in `reprojex/CONSUMER_CONTRACT.md`. This
//! module owns only the parts reprojex dictates; xrizer policy (staleness
//! cutoff, logging, swapchain/import design) lives with callers.
//!
//! ```text
//! +0  u32 magic       0x58504450 ('XPDP')
//! +4  u32 version     3
//! +8  u32 seq         seqlock: odd = writer inside
//! +12 u32 w, +16 u32 h   full side-by-side mirror dims (eye_w = w/2)
//! +20 i32 format      126 = D32F (preferred), 130 = D32S8 (fallback)
//! +24 u32 epoch       bumped per mirror rebuild; reimport all FDs on change
//! +28 u32 reserved0   zero
//! +32 u64 generation  latest completed generation (0 = nothing published)
//! +40 u64 time_mono_ns  CLOCK_MONOTONIC at copy completion
//! +48 i32 mem_fds[2], +56 i32 sem_fds[2]  (-1 until published, valid only > 2)
//! +64 u32 alloc_bytes import size must be <= this; 0 = unknown
//! +68 f32 near_z, +72 f32 far_z  clip planes
//! +76 u32 reserved_pad  keeps pinned 8-byte aligned as a pair
//! +80 u32 pinned[2]   consumer-written only, OUTSIDE the seqlock
//! +88 u32 reserved[2] zero
//! ```
//!
//! Reader discipline: fresh `shm_open`+`mmap` per call (producer
//! unlink/recreates on geometry change), `fstat` size must be exactly 96,
//! seqlock retry on odd/changed `seq`, `generation == 0` means nothing
//! published yet. `pinned[]` are single atomic words written only by the
//! consumer (`Release` store 1 before wait/sample, 0 after own fence) and
//! read by the producer; never read them inside the seqlock snapshot.
//! Slot for generation `G` is `G % 2`. Semaphores are timeline semaphores
//! signaled with value `G` and must be imported as timeline type.
//! `CLOCK_MONOTONIC` timestamps correlate frames: `age = receipt - header`.
//!
//! No new crates: `libc` + `log` only.

// Slice 1 is wire-only with no callers yet; compositor/import slices will
// consume every helper below. Silence dead-code until then.
#![allow(dead_code)]

use log::{debug, info, warn};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Fixed rendezvous. No env override: a rendezvous point must not be configurable.
pub const SHM_NAME: &str = "/xplane-depth-v3";
pub const SHM_MAGIC: u32 = 0x5850_4450; // 'XPDP'
pub const SHM_VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 96;
pub const SLOTS: usize = 2;
/// Byte offset of `pinned[0]`. Both words are consumer-written atomics.
pub const PINNED_OFF: usize = 80;
/// Vulkan `VK_FORMAT_D32_SFLOAT`. Preferred depth format in prod.
pub const FORMAT_D32F: i32 = 126;
/// Vulkan `VK_FORMAT_D32_SFLOAT_S8_UINT`. Accepted fallback. Mirror/copy
/// are depth-aspect-only, so stencil never enters the feed.
pub const FORMAT_D32S8: i32 = 130;
/// Fixed depth mapping range. v3 carries no min/max fields; always [0, 1].
pub const MIN_DEPTH: f32 = 0.0;
pub const MAX_DEPTH: f32 = 1.0;
/// Default staleness bound (ms) when `XRIZER_DEPTH_MAX_AGE_MS` is unset.
pub const DEFAULT_MAX_AGE_MS: u64 = 100;

/// Mirror of the C `rx_shm_v3`. Must stay 96 bytes with the exact offsets
/// below. Keep in sync with `reprojex/src/shm.h`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct XdShmHeaderV3 {
    pub magic: u32,         // +0
    pub version: u32,       // +4
    pub seq: u32,           // +8
    pub w: u32,             // +12
    pub h: u32,             // +16
    pub format: i32,        // +20
    pub epoch: u32,         // +24
    pub reserved0: u32,     // +28
    pub generation: u64,    // +32
    pub time_mono_ns: u64,  // +40
    pub mem_fds: [i32; 2],  // +48
    pub sem_fds: [i32; 2],  // +56
    pub alloc_bytes: u32,   // +64
    pub near_z: f32,        // +68
    pub far_z: f32,         // +72
    pub reserved_pad: u32,  // +76
    pub pinned: [u32; 2],   // +80 consumer-written, outside seqlock
    pub reserved: [u32; 2], // +88
}

// Pinned at build time, not just under test: a drifted offset is a
// cross-process ABI break, not a test failure.
const _: () = {
    assert!(std::mem::size_of::<XdShmHeaderV3>() == HEADER_SIZE);
    assert!(std::mem::offset_of!(XdShmHeaderV3, magic) == 0);
    assert!(std::mem::offset_of!(XdShmHeaderV3, version) == 4);
    assert!(std::mem::offset_of!(XdShmHeaderV3, seq) == 8);
    assert!(std::mem::offset_of!(XdShmHeaderV3, w) == 12);
    assert!(std::mem::offset_of!(XdShmHeaderV3, h) == 16);
    assert!(std::mem::offset_of!(XdShmHeaderV3, format) == 20);
    assert!(std::mem::offset_of!(XdShmHeaderV3, epoch) == 24);
    assert!(std::mem::offset_of!(XdShmHeaderV3, reserved0) == 28);
    assert!(std::mem::offset_of!(XdShmHeaderV3, generation) == 32);
    assert!(std::mem::offset_of!(XdShmHeaderV3, time_mono_ns) == 40);
    assert!(std::mem::offset_of!(XdShmHeaderV3, mem_fds) == 48);
    assert!(std::mem::offset_of!(XdShmHeaderV3, sem_fds) == 56);
    assert!(std::mem::offset_of!(XdShmHeaderV3, alloc_bytes) == 64);
    assert!(std::mem::offset_of!(XdShmHeaderV3, near_z) == 68);
    assert!(std::mem::offset_of!(XdShmHeaderV3, far_z) == 72);
    assert!(std::mem::offset_of!(XdShmHeaderV3, reserved_pad) == 76);
    assert!(std::mem::offset_of!(XdShmHeaderV3, pinned) == PINNED_OFF);
    assert!(std::mem::offset_of!(XdShmHeaderV3, reserved) == 88);
};

/// Snapshot of the v3 control header. `pinned[]` is deliberately absent:
/// those words live outside the seqlock and must be touched only through
/// [`DepthFeed::pin_slot`]/[`DepthFeed::unpin_slot`]/[`DepthFeed::load_pin`].
#[derive(Debug, Clone)]
pub struct DepthControl {
    pub epoch: u32,
    pub w: u32,
    pub h: u32,
    pub format: i32,
    /// Latest completed generation (`generation`; 0 = nothing published).
    pub generation: u64,
    pub time_ns: u64,
    pub mem_fds: [i32; 2],
    pub sem_fds: [i32; 2],
    pub near_z: f32,
    pub far_z: f32,
    /// Observed segment length; viable only when exactly 96.
    pub seg_len: usize,
    /// Driver-reported mirror allocation size (+64). Import size must be
    /// `<= alloc_bytes` when non-zero; 0 = unknown (tolerate).
    pub alloc_bytes: u32,
}

impl DepthControl {
    /// Slot holding `generation`. Contract: `G % 2`, no header field.
    pub fn slot_for_generation(generation: u64) -> usize {
        (generation % SLOTS as u64) as usize
    }

    /// Slot holding this snapshot's generation.
    pub fn slot(&self) -> usize {
        Self::slot_for_generation(self.generation)
    }

    /// Single-eye width of the side-by-side mirror.
    pub fn eye_w(&self) -> u32 {
        self.w / 2
    }

    /// Full-mirror texel footprint for `bpp` bytes/texel.
    pub fn want_bytes(&self, bpp: u64) -> u64 {
        u64::from(self.w) * u64::from(self.h) * bpp
    }

    /// `alloc_bytes` bound: `want > alloc` (non-zero) means writer/reader
    /// drift; reject before any import call. `alloc == 0` tolerates old
    /// producers that predate the field.
    pub fn alloc_covers(&self, bpp: u64) -> bool {
        self.alloc_bytes == 0 || self.want_bytes(bpp) <= u64::from(self.alloc_bytes)
    }

    /// All four FDs must be real (stdin/stdout/stderr are never depth).
    /// The producer's pre-publish sentinel is -1.
    pub fn fds_valid(&self) -> bool {
        self.mem_fds.iter().all(|&f| f > 2) && self.sem_fds.iter().all(|&f| f > 2)
    }

    pub fn dims_plausible(&self) -> bool {
        self.w != 0 && self.h != 0 && self.w <= 16384 && self.h <= 16384 && self.w.is_multiple_of(2)
    }

    /// Full viability gate: v3, published epoch+generation, depth SBS
    /// (D32F preferred, D32S8 fallback), plausible dims, exact 96-byte
    /// segment, all FDs real. `near/far` validity is checked separately
    /// via [`validate_near_far`].
    pub fn is_viable(&self) -> bool {
        self.epoch != 0
            && self.generation != 0
            && (self.format == FORMAT_D32F || self.format == FORMAT_D32S8)
            && self.seg_len == HEADER_SIZE
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

    /// Signed age of this snapshot at Submit receipt: positive = header
    /// older than receipt (normal), negative = header newer (same frame
    /// arriving just after wakeup; Submit-before-Present ordering).
    pub fn age_ns(&self, receipt_mono_ns: u64) -> i64 {
        depth_age_ns(receipt_mono_ns, self.time_ns)
    }
}

/// Reader handle. Stateless across frames (open/mmap per call) so producer
/// `munmap`/`shm_unlink`/recreate cycles on geometry changes are picked up
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
        // Per-frame when the layer is absent: throttle so RUST_LOG=debug
        // stays readable (one line per 5 s, not ~90/s).
        static LAST_ABSENT_LOG_NS: AtomicU64 = AtomicU64::new(0);
        let now = now_mono_ns();
        if now.saturating_sub(LAST_ABSENT_LOG_NS.load(Ordering::Relaxed)) > 5_000_000_000 {
            LAST_ABSENT_LOG_NS.store(now, Ordering::Relaxed);
            debug!("depth feed: shm_open failed (absent)");
        }
        return None;
    }
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        unsafe { libc::close(fd) };
        return None;
    }
    let map_len = stat.st_size as usize;
    // v3 needs exactly the 96-byte header: anything else belongs to a dead
    // run (unlinked, mid-recreate) or a foreign writer.
    if map_len != HEADER_SIZE {
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
            info!("depth feed: SHM path {SHM_NAME} (master on)");
        }
        Some(Self {
            name: CString::new(SHM_NAME).expect("SHM name contains NUL"),
        })
    }

    /// Test seam: custom segment name (prod path is always fixed).
    #[cfg(test)]
    fn open_for_test(name: &str) -> Self {
        Self {
            name: CString::new(name).expect("SHM name contains NUL"),
        }
    }

    /// Seqlock-read of the v3 control header. Returns `None` on
    /// absent/wrong-size/torn/foreign data so callers fall back to plain
    /// (color-only) views. Never touches `pinned[]`: those words are
    /// outside the seqlock (see [`Self::pin_slot`]).
    pub fn read_control(&self) -> Option<DepthControl> {
        let map = map_segment(&self.name, false)?;
        let (ptr, map_len) = (map.ptr, map.len);
        debug_assert_eq!(map_len, HEADER_SIZE);
        for _ in 0..4 {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            if unsafe { read_u32(ptr, 0) } != SHM_MAGIC {
                return None;
            }
            if unsafe { read_u32(ptr, 4) } != SHM_VERSION {
                return None;
            }
            // Copy under seq1; the closing seq check covers torn reads.
            let h = unsafe { read_header_v3(ptr) };
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            if h.magic != SHM_MAGIC || h.version != SHM_VERSION {
                return None;
            }
            return Some(DepthControl {
                epoch: h.epoch,
                w: h.w,
                h: h.h,
                format: h.format,
                generation: h.generation,
                time_ns: h.time_mono_ns,
                mem_fds: h.mem_fds,
                sem_fds: h.sem_fds,
                near_z: h.near_z,
                far_z: h.far_z,
                seg_len: map_len,
                alloc_bytes: h.alloc_bytes,
            });
        }
        None
    }

    fn pin_addr(map: &MappedSeg, slot: usize) -> Option<*mut AtomicU32> {
        if slot >= SLOTS || map.len != HEADER_SIZE {
            return None;
        }
        // Release assert: the AtomicU32 cast below faults on misalignment
        // on strict architectures; page mappings are aligned in practice,
        // but never skip this check in release.
        assert_eq!(
            (map.ptr as usize + PINNED_OFF + slot * 4) % std::mem::align_of::<AtomicU32>(),
            0
        );
        Some(unsafe {
            (map.ptr as *mut u8)
                .add(PINNED_OFF + slot * 4)
                .cast::<AtomicU32>()
        })
    }

    /// Pin `slot` before waiting/sampling it (`Release` store of 1).
    /// At most one slot may be pinned at a time; holding both stalls the
    /// producer. Returns false when the segment is absent or `slot >= 2`.
    pub fn pin_slot(&self, slot: usize) -> bool {
        let Some(map) = map_segment(&self.name, true) else {
            return false;
        };
        let Some(addr) = Self::pin_addr(&map, slot) else {
            return false;
        };
        unsafe {
            if let Some(a) = addr.as_ref() {
                a.store(1, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Unpin `slot` after the sampler's own fence confirms completion
    /// (not after submit). Returns false when absent or out of range.
    pub fn unpin_slot(&self, slot: usize) -> bool {
        let Some(map) = map_segment(&self.name, true) else {
            return false;
        };
        let Some(addr) = Self::pin_addr(&map, slot) else {
            return false;
        };
        unsafe {
            if let Some(a) = addr.as_ref() {
                a.store(0, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Single `Acquire` load of a pin word. `None` when absent/out of range.
    pub fn load_pin(&self, slot: usize) -> Option<u32> {
        let map = map_segment(&self.name, false)?;
        let addr = Self::pin_addr(&map, slot)?;
        unsafe { addr.as_ref().map(|a| a.load(Ordering::Acquire)) }
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

unsafe fn read_header_v3(base: *mut libc::c_void) -> XdShmHeaderV3 {
    // Same: torn reads are detected via the seqlock retry, but the read
    // itself must be volatile to be sound under concurrent mutation.
    unsafe { (base as *const XdShmHeaderV3).read_volatile() }
}

/// Header near/far win — no xrizer-side defaults. Pure predicate: invalid
/// (`<= 0`, non-finite, `near >= far`) yields `None` so the caller falls
/// back to plain views; the caller owns logging.
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

/// Signed frame age: `receipt - header`, both `CLOCK_MONOTONIC` ns.
/// Positive = depth older than the Submit wakeup (normal); negative =
/// depth newer than the wakeup (same frame arriving just after; the
/// producer publishes at present, after Submit is woken).
pub fn depth_age_ns(receipt_mono_ns: u64, header_mono_ns: u64) -> i64 {
    if receipt_mono_ns >= header_mono_ns {
        (receipt_mono_ns - header_mono_ns).min(i64::MAX as u64) as i64
    } else {
        -((header_mono_ns - receipt_mono_ns).min(i64::MAX as u64) as i64)
    }
}

/// Same-generation rule from the wire contract: `-margin <= age <
/// period + margin`. A timestamp inside the last frame period before
/// Submit is that frame's depth; a marginally newer stamp is the same
/// frame arriving just after wakeup — freshest case, not an error.
pub fn is_same_generation(age_ns: i64, frame_period_ns: i64, margin_ns: i64) -> bool {
    age_ns >= -margin_ns && age_ns < frame_period_ns.saturating_add(margin_ns)
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

    fn make_seg_v3(
        name: &str,
        w: u32,
        h: u32,
        generation: u64,
        tweak: impl FnOnce(&mut XdShmHeaderV3),
    ) -> TestSeg {
        let cname = CString::new(name).unwrap();
        unsafe { libc::shm_unlink(cname.as_ptr()) };
        let len = HEADER_SIZE;
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
        let mut hdr = XdShmHeaderV3 {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            seq: 1,
            w,
            h,
            format: FORMAT_D32F,
            epoch: 7,
            reserved0: 0,
            generation,
            time_mono_ns: now_mono_ns(),
            mem_fds: [11, 12],
            sem_fds: [21, 22],
            alloc_bytes: 8 * 2 * 4,
            near_z: 0.00742,
            far_z: 10000.0,
            reserved_pad: 0,
            pinned: [0, 0],
            reserved: [0, 0],
        };
        tweak(&mut hdr);
        unsafe {
            (ptr as *mut XdShmHeaderV3).write_unaligned(hdr);
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
    fn header_layout_matches_wire_contract() {
        // Byte-identical with reprojex/src/shm.h (v3, 96B).
        assert_eq!(std::mem::size_of::<XdShmHeaderV3>(), 96);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, magic), 0);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, version), 4);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, seq), 8);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, w), 12);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, h), 16);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, format), 20);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, epoch), 24);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, reserved0), 28);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, generation), 32);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, time_mono_ns), 40);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, mem_fds), 48);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, sem_fds), 56);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, alloc_bytes), 64);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, near_z), 68);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, far_z), 72);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, reserved_pad), 76);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, pinned), PINNED_OFF);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV3, reserved), 88);
    }

    #[test]
    fn roundtrip_viable() {
        let name = format!("/xrizer-depth-test-{}-1", std::process::id());
        let _seg = make_seg_v3(&name, 8, 2, 42, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("should hit");
        assert_eq!((ctl.w, ctl.h, ctl.generation), (8, 2, 42));
        assert_eq!(ctl.epoch, 7);
        assert_eq!(ctl.mem_fds, [11, 12]);
        assert_eq!(ctl.sem_fds, [21, 22]);
        assert_eq!(ctl.alloc_bytes, 8 * 2 * 4);
        assert!((ctl.near_z - 0.00742).abs() < f32::EPSILON);
        assert!((ctl.far_z - 10000.0).abs() < f32::EPSILON);
        assert!(ctl.is_viable());
        assert_eq!(ctl.eye_w(), 4);
        assert_eq!(ctl.slot(), 0); // 42 % 2
        assert!(!ctl.is_stale(now_mono_ns(), 100));
        assert!(ctl.alloc_covers(4));
        assert!(!ctl.alloc_covers(4 * 1000));
    }

    #[test]
    fn slot_mapping_is_gen_mod_two() {
        assert_eq!(DepthControl::slot_for_generation(0), 0);
        assert_eq!(DepthControl::slot_for_generation(1), 1);
        assert_eq!(DepthControl::slot_for_generation(2), 0);
        assert_eq!(DepthControl::slot_for_generation(43), 1);
    }

    #[test]
    fn old_versions_rejected() {
        let pid = std::process::id();
        // 128-byte v2-shaped segment: wrong size entirely.
        let name = format!("/xrizer-depth-test-{pid}-v2a");
        let _seg = make_raw_seg(&name, 128, |ptr| unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0, 128);
            (ptr as *mut u32).write_volatile(SHM_MAGIC);
            (ptr as *mut u32).add(1).write_volatile(2);
            (ptr as *mut u32).add(2).write_volatile(2);
        });
        assert!(
            DepthFeed::open_for_test(&name).read_control().is_none(),
            "v2 size must miss"
        );
        // 96-byte segment stamped version 2: wrong version.
        let name2 = format!("/xrizer-depth-test-{pid}-v2b");
        let _seg2 = make_seg_v3(&name2, 8, 2, 1, |h| {
            h.version = 2;
        });
        assert!(
            DepthFeed::open_for_test(&name2).read_control().is_none(),
            "version 2 must miss"
        );
    }

    #[test]
    fn bad_magic_and_size_rejected() {
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-mag");
        let seg = make_seg_v3(&name, 8, 2, 1, |_| {});
        unsafe {
            (seg.ptr as *mut u32).write_volatile(0xdead_beef);
            (seg.ptr as *mut u32).add(2).write_volatile(2);
        }
        assert!(
            DepthFeed::open_for_test(&name).read_control().is_none(),
            "bad magic must miss"
        );
        // Oversized segment: foreign writer, not our 96B header.
        let name2 = format!("/xrizer-depth-test-{pid}-big");
        let _seg2 = make_raw_seg(&name2, HEADER_SIZE + 32, |ptr| unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0, HEADER_SIZE + 32);
            (ptr as *mut u32).write_volatile(SHM_MAGIC);
            (ptr as *mut u32).add(1).write_volatile(SHM_VERSION);
            (ptr as *mut u32).add(2).write_volatile(2);
        });
        assert!(
            DepthFeed::open_for_test(&name2).read_control().is_none(),
            "wrong size must miss"
        );
    }

    #[test]
    fn sentinel_and_low_fds_rejected() {
        let pid = std::process::id();
        let cases: [(i32, i32); 5] = [(-1, 21), (0, 21), (1, 21), (2, 21), (11, 2)];
        for (i, (mem0, sem0)) in cases.iter().enumerate() {
            let name = format!("/xrizer-depth-test-{pid}-fd{i}");
            let _seg = make_seg_v3(&name, 8, 2, 3, |h| {
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
        let _seg = make_seg_v3(&name, 8, 2, 0, |h| {
            h.epoch = 0;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "epoch/gen 0 must not arm");
        // Bad dims / format.
        let name = format!("/xrizer-depth-test-{pid}-oddw");
        let _seg = make_seg_v3(&name, 7, 2, 3, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "odd width must not arm");
        type Tweak = (&'static str, fn(&mut XdShmHeaderV3));
        let tweaks: [Tweak; 5] = [
            ("zerow", |h: &mut XdShmHeaderV3| h.w = 0),
            ("huge", |h: &mut XdShmHeaderV3| h.w = 16385),
            ("fmt", |h: &mut XdShmHeaderV3| h.format = 96),
            ("gen0", |h: &mut XdShmHeaderV3| h.generation = 0),
            ("epoch0", |h: &mut XdShmHeaderV3| h.epoch = 0),
        ];
        for (tag, tweak) in tweaks {
            let name = format!("/xrizer-depth-test-{pid}-{tag}");
            let _seg = make_seg_v3(&name, 8, 2, 3, tweak);
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("readable");
            assert!(!ctl.is_viable(), "{tag} must not arm");
        }
    }

    #[test]
    fn d32s8_fallback_is_viable() {
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-s8");
        let _seg = make_seg_v3(&name, 8, 2, 3, |h| {
            h.format = FORMAT_D32S8;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert_eq!(ctl.format, FORMAT_D32S8);
        assert!(ctl.is_viable(), "D32S8 fallback must arm");
    }

    #[test]
    fn stale_and_future_are_stale() {
        let pid = std::process::id();
        let now = now_mono_ns();
        let name = format!("/xrizer-depth-test-{pid}-stale");
        let _seg = make_seg_v3(&name, 8, 2, 7, |h| {
            h.time_mono_ns = now.saturating_sub(500_000_000);
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable());
        assert!(ctl.is_stale(now, 100), "500ms old must be stale");
        let name2 = format!("/xrizer-depth-test-{pid}-future");
        let _seg2 = make_seg_v3(&name2, 8, 2, 8, |h| {
            h.time_mono_ns = now.saturating_add(50_000_000);
        });
        let ctl2 = DepthFeed::open_for_test(&name2)
            .read_control()
            .expect("readable");
        assert!(ctl2.is_stale(now, 100), "future stamp must be stale");
    }

    #[test]
    fn pin_roundtrip_outside_seqlock() {
        let name = format!("/xrizer-depth-test-{}-pin", std::process::id());
        let seg = make_seg_v3(&name, 8, 2, 42, |_| {});
        let feed = DepthFeed::open_for_test(&name);
        // Fresh segment boots unpinned; snapshot reads fine alongside pins.
        assert_eq!(feed.load_pin(0), Some(0));
        assert_eq!(feed.load_pin(1), Some(0));
        assert!(feed.read_control().is_some());
        assert!(feed.pin_slot(0));
        assert_eq!(feed.load_pin(0), Some(1));
        assert_eq!(feed.load_pin(1), Some(0));
        // Snapshot still clean while pinned (pins are outside the seqlock).
        assert!(feed.read_control().is_some());
        assert!(feed.unpin_slot(0));
        assert_eq!(feed.load_pin(0), Some(0));
        // Out-of-range slots never touch the mapping.
        assert!(!feed.pin_slot(2));
        assert!(!feed.unpin_slot(2));
        assert_eq!(feed.load_pin(2), None);
        // Direct check the words sit at +80/+84.
        unsafe {
            assert_eq!((seg.ptr as *const u8).add(80).cast::<u32>().read(), 0);
            assert_eq!((seg.ptr as *const u8).add(84).cast::<u32>().read(), 0);
        }
    }

    #[test]
    fn stuck_odd_seqlock_returns_none() {
        let name = format!("/xrizer-depth-test-{}-odd", std::process::id());
        let seg = make_seg_v3(&name, 8, 2, 1, |_| {});
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
        assert!(!feed.pin_slot(0));
        assert!(!feed.unpin_slot(0));
        assert_eq!(feed.load_pin(0), None);
    }

    #[test]
    fn near_far_strict() {
        assert_eq!(
            validate_near_far(0.00742, 10000.0),
            Some((0.00742, 10000.0))
        );
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
    fn signed_age_and_same_generation() {
        // Older header: positive age.
        assert_eq!(depth_age_ns(1_000, 900), 100);
        // Newer header (straddled present): negative age.
        assert_eq!(depth_age_ns(1_000, 1_002), -2);
        // Same-generation window: last period plus small negative margin.
        assert!(is_same_generation(5_000_000, 11_111_111, 2_000_000));
        assert!(is_same_generation(-1_000_000, 11_111_111, 2_000_000));
        assert!(!is_same_generation(-3_000_000, 11_111_111, 2_000_000));
        assert!(!is_same_generation(15_000_000, 11_111_111, 2_000_000));
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
