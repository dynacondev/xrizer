//! SHM depth-feed reader: xrizer side of the reprojex v4 handoff.
//!
//! Wire contract lives in `reprojex/src/shm.h` (`/xplane-depth-v4`, 96
//! bytes) with reader ordering in `reprojex/CONSUMER_CONTRACT.md`. This
//! module owns only the parts reprojex dictates; xrizer policy (association
//! window, logging, swapchain/import design) lives with callers.
//!
//! ```text
//! +0  u32 magic       0x58504450 ('XPDP')
//! +4  u32 version     4
//! +8  u32 seq         seqlock: odd = writer inside
//! +12 u32 w, +16 u32 h   full side-by-side mirror dims (eye_w = w/2)
//! +20 i32 format      126 = D32F (preferred), 130 = D32S8 (fallback)
//! +24 u32 epoch       bumped per mirror rebuild; reimport all FDs on change
//! +28 u32 reserved0   zero
//! +32 u64 generation  latest completed generation (0 = nothing published)
//! +40 u64 time_mono_ns  CLOCK_MONOTONIC at copy completion
//! +48 i32 mem_fds[2]    dma-buf memory, imported once per epoch
//! +56 i32 fence_fds[2]  one-shot sync_file snapshot per publish
//!                         (-1 until published, valid only > 2)
//! +64 u32 alloc_bytes import size must be <= this; 0 = unknown
//! +68 f32 near_z, +72 f32 far_z  clip planes
//! +76 u32 reserved_pad  keeps pinned 8-byte aligned as a pair
//! +80 u32 pinned[2]   consumer-written only, OUTSIDE the seqlock
//! +88 u32 reserved[2] zero
//! ```
//!
//! Reader discipline: fresh `shm_open`+`mmap` per call (producer
//! unlink/recreates on geometry change, so no fd is ever cached), `fstat`
//! size must be exactly 96, seqlock retry (bounded, [`READ_RETRIES`]
//! tries) on odd/changed `seq`, `generation == 0` means nothing
//! published yet. `read_control` returns [`ReadSkip`] on every miss so the
//! compositor heartbeat can bucket skips with counter increments only —
//! no formatting in the loop; the mapping itself plus one `clock_gettime`
//! per miss for throttle gating are the only syscalls. The wire module
//! logs only caller-invisible states (absent segment, exhausted retries),
//! throttled to 1/5s at `debug!`; everything else is a returned reason.
//! `pinned[]` are single atomic words written only by the consumer
//! (`Release` store 1 before poll/sample, 0 after own fence) and read by
//! the producer; the logical snapshot excludes them (the physical copy
//! reads the whole mapping but discards the pin words — they are outside
//! the seqlock).
//! At most one pin at a time — the API does not enforce this, the caller
//! must; holding both stalls all publishes. Pins are backpressure-skip
//! hints, not acks: the producer skips a pinned slot and keeps the other
//! moving, so a stuck pin costs half rate, never a wedge.
//! Slot for generation `G` is `G % 2`. Per-slot completion is a
//! one-shot fence sync_file, exported fresh at every publish: poll it
//! ([`poll_fence_fd`], bounded) instead of any semaphore wait — there is
//! no semaphore object to import, cache, or re-wait, so skipped
//! generations leave no sync debt behind by construction.
//! `CLOCK_MONOTONIC` timestamps correlate frames: `age = receipt - header`.
//!
//! FDs are borrowed numbers used directly (no `SCM_RIGHTS`/re-open/close,
//! never closed by the reader); valid only `> 2`, `-1` until published.
//! Memory FDs are (re)imported on epoch change. Fence FDs are single-use
//! snapshots: always use the current read's numbers, never a cached copy
//! across frames — the producer closes and replaces a slot's fence FD on
//! reuse, and reuse is gated on the slot being unpinned, so a held pin
//! keeps polled numbers stable. `reserved0/reserved_pad/reserved`
//! must be tolerated nonzero (forward-compat; never checked). Viability
//! is `is_viable()` AND `alloc_covers(bpp)` AND `validate_near_far()` —
//! none implies the others. `eye_w() == w/2` truncates; odd `w` is rejected
//! by `dims_plausible`, never rounded.
//! The reader never `shm_unlink`s/truncates/writes the header except
//! `pinned[]`; mappings are `munmap`ed+closed on drop with no caching, and
//! disappearance is treated like generation 0 (drop imports, plain frame,
//! re-open).
//!
//! Submission gating is [`is_same_generation`] on the signed
//! frame-vs-depth age — there is deliberately no absolute-age "stale"
//! gate. A tight per-frame window (period + small margin) subsumes
//! liveness: same-generation success implies fresh, and a frozen
//! `generation` across frames (not a clock comparison) is the stalled
//! producer signal. Feed [`AgeStats`] with one [`depth_age_ns`] sample per
//! successful read only — misses carry no timestamp, so they are counted
//! via [`ReadSkip`] buckets instead — and report it on the heartbeat to
//! catch sync drift.
//!
//! Removed knobs: `XRIZER_DEPTH_MAX_AGE_MS` (absolute-age gate) no longer
//! exists — if set it is ignored; the kill-switch `XRIZER_DEPTH_SHM` is
//! the only depth env var.
//!
//! No new crates: `libc` + `log` only.

use log::{debug, info, warn};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Fixed rendezvous. No env override: a rendezvous point must not be configurable.
pub const SHM_NAME: &str = "/xplane-depth-v4";
pub const SHM_MAGIC: u32 = 0x5850_4450; // 'XPDP'
pub const SHM_VERSION: u32 = 4;
pub const HEADER_SIZE: usize = 96;
pub const SLOTS: usize = 2;
/// Byte offset of `pinned[0]`. Both words are consumer-written atomics.
pub const PINNED_OFF: usize = 80;
/// Vulkan `VK_FORMAT_D32_SFLOAT`. Preferred depth format in prod.
pub const FORMAT_D32F: i32 = 126;
/// Vulkan `VK_FORMAT_D32_SFLOAT_S8_UINT`. Accepted fallback. Mirror/copy
/// are depth-aspect-only, so stencil never enters the feed.
pub const FORMAT_D32S8: i32 = 130;
/// Fixed depth mapping range. v4 carries no min/max fields; always [0, 1].
pub const MIN_DEPTH: f32 = 0.0;
pub const MAX_DEPTH: f32 = 1.0;
/// Bounded seqlock retries in [`DepthFeed::read_control`]. The writer holds
/// odd for one 96-byte fill (nanoseconds, no syscalls inside), so a healthy
/// read succeeds on try 1; the bound exists so a dead/stuck producer
/// degrades to a [`ReadSkip::Torn`] (plain frame) instead of spinning the
/// compositor thread. Named so the tradeoff is reviewable.
pub const READ_RETRIES: usize = 4;

/// Mirror of the C `rx_shm_v4`. Must stay 96 bytes with the exact offsets
/// below. Keep in sync with `reprojex/src/shm.h`. Offsets are unchanged
/// from v3; only `version` and the `fence_fds` semantics changed (v3 used
/// timeline semaphores, v4 exports one-shot fence sync_files).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct XdShmHeaderV4 {
    pub magic: u32,          // +0
    pub version: u32,        // +4
    pub seq: u32,            // +8
    pub w: u32,              // +12
    pub h: u32,              // +16
    pub format: i32,         // +20
    pub epoch: u32,          // +24
    pub reserved0: u32,      // +28
    pub generation: u64,     // +32
    pub time_mono_ns: u64,   // +40
    pub mem_fds: [i32; 2],   // +48 dma-buf memory
    pub fence_fds: [i32; 2], // +56 one-shot sync_file per publish
    pub alloc_bytes: u32,    // +64
    pub near_z: f32,         // +68
    pub far_z: f32,          // +72
    pub reserved_pad: u32,   // +76
    pub pinned: [u32; 2],    // +80 consumer-written, outside seqlock
    pub reserved: [u32; 2],  // +88
}

// Pinned at build time, not just under test: a drifted offset is a
// cross-process ABI break, not a test failure.
const _: () = {
    assert!(std::mem::size_of::<XdShmHeaderV4>() == HEADER_SIZE);
    assert!(std::mem::offset_of!(XdShmHeaderV4, magic) == 0);
    assert!(std::mem::offset_of!(XdShmHeaderV4, version) == 4);
    assert!(std::mem::offset_of!(XdShmHeaderV4, seq) == 8);
    assert!(std::mem::offset_of!(XdShmHeaderV4, w) == 12);
    assert!(std::mem::offset_of!(XdShmHeaderV4, h) == 16);
    assert!(std::mem::offset_of!(XdShmHeaderV4, format) == 20);
    assert!(std::mem::offset_of!(XdShmHeaderV4, epoch) == 24);
    assert!(std::mem::offset_of!(XdShmHeaderV4, reserved0) == 28);
    assert!(std::mem::offset_of!(XdShmHeaderV4, generation) == 32);
    assert!(std::mem::offset_of!(XdShmHeaderV4, time_mono_ns) == 40);
    assert!(std::mem::offset_of!(XdShmHeaderV4, mem_fds) == 48);
    assert!(std::mem::offset_of!(XdShmHeaderV4, fence_fds) == 56);
    assert!(std::mem::offset_of!(XdShmHeaderV4, alloc_bytes) == 64);
    assert!(std::mem::offset_of!(XdShmHeaderV4, near_z) == 68);
    assert!(std::mem::offset_of!(XdShmHeaderV4, far_z) == 72);
    assert!(std::mem::offset_of!(XdShmHeaderV4, reserved_pad) == 76);
    assert!(std::mem::offset_of!(XdShmHeaderV4, pinned) == PINNED_OFF);
    assert!(std::mem::offset_of!(XdShmHeaderV4, pinned) + 4 == 84);
    assert!(std::mem::offset_of!(XdShmHeaderV4, reserved) == 88);
};

/// Snapshot of the v4 control header. `pinned[]` is deliberately absent:
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
    /// One-shot fence sync_file for this generation's copy completion.
    /// Use only the current read's numbers: the producer closes and replaces
    /// a slot's fence FD on reuse (gated on the slot being unpinned), so a
    /// cached copy across frames may name a closed or recycled FD. The held
    /// pin keeps the polled numbers stable for this frame.
    pub fence_fds: [i32; 2],
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

    /// Full-mirror texel footprint for `bpp` bytes/texel. Saturates:
    /// `w/h` are untrusted pre-gate values, so plain multiplication could
    /// wrap small and bypass [`Self::alloc_covers`].
    pub fn want_bytes(&self, bpp: u64) -> u64 {
        u64::from(self.w)
            .saturating_mul(u64::from(self.h))
            .saturating_mul(bpp)
    }

    /// `alloc_bytes` bound: `want > alloc` (non-zero) means writer/reader
    /// drift; reject before any import call. `alloc == 0` tolerates old
    /// producers that predate the field.
    pub fn alloc_covers(&self, bpp: u64) -> bool {
        self.alloc_bytes == 0 || self.want_bytes(bpp) <= u64::from(self.alloc_bytes)
    }

    /// All four FDs must be real (stdin/stdout/stderr are never depth).
    /// The producer's pre-publish sentinel is -1. Memory FDs persist per
    /// epoch; fence FDs are single-use snapshots (see field docs).
    /// Intentionally strict (first publish after a build leaves the idle
    /// slot at -1, forcing one plain frame): per-slot viability would arm
    /// a frame earlier at the cost of a second gate shape for zero safety
    /// gain — the snapshot's other-slot numbers are meaningless for the
    /// sampled generation anyway.
    pub fn fds_valid(&self) -> bool {
        self.mem_fds.iter().all(|&f| f > 2) && self.fence_fds.iter().all(|&f| f > 2)
    }

    /// Reader-policy gate: even `w` required for the SBS split. The wire
    /// contract implies SBS but never states evenness; the mirror publishes
    /// whatever the app's depth is, so odd `w` is rejected here (fail-safe
    /// to plain), not claimed as unwritable.
    pub fn dims_plausible(&self) -> bool {
        // NOTE: `% 2` instead of `is_multiple_of` — workspace MSRV is 1.85.
        #[allow(clippy::manual_is_multiple_of)]
        let even = self.w % 2 == 0;
        self.w != 0 && self.h != 0 && self.w <= 16384 && self.h <= 16384 && even
    }

    /// Full viability gate: published epoch+generation, depth format
    /// (D32F preferred, D32S8 fallback per `reprojex/src/catalog.c`), plausible
    /// dims, exact 96-byte segment, all FDs real. `epoch != 0` has no
    /// contract sentence; it holds because the producer zero-inits the
    /// mapping and its epoch is process-monotonic starting at 1. `alloc`
    /// and `near/far` are deliberately excluded — callers must AND
    /// [`Self::alloc_covers`] and [`validate_near_far`].
    pub fn is_viable(&self) -> bool {
        self.epoch != 0
            && self.generation != 0
            && (self.format == FORMAT_D32F || self.format == FORMAT_D32S8)
            && self.seg_len == HEADER_SIZE
            && self.dims_plausible()
            && self.fds_valid()
    }

    /// Signed age of this snapshot at end_frame-entry receipt: positive
    /// = header older than receipt (normal: present landed just before the
    /// handoff), negative = header newer (same frame arriving just after
    /// wakeup; Submit-before-Present ordering). Feed every successful read
    /// into [`AgeStats`]; read misses have no timestamp and are bucketed
    /// by [`ReadSkip`] instead.
    pub fn age_ns(&self, receipt_mono_ns: u64) -> i64 {
        depth_age_ns(receipt_mono_ns, self.time_ns)
    }
}

/// Why a read missed, for heartbeat bucketing. `Copy` enum only — no
/// strings, no formatting — so the hot path pays a discriminant move and
/// the compositor aggregates counters for its periodic heartbeat line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadSkip {
    /// Segment absent (open/fstat/mmap failed) — no layer installed, torn
    /// down, or transient resource failure (indistinguishable by design;
    /// OOM buckets here too). Throttled `debug!` fires only on the
    /// `shm_open`-absent path; post-open failures are counter-only.
    Absent,
    /// Mapped but not exactly 96 bytes — foreign writer or mid-recreate.
    WrongSize,
    /// Seqlock never settled within [`READ_RETRIES`] (stuck odd or
    /// repeated torn copies). Throttled `debug!` at exhaustion.
    Torn,
    /// Magic mismatch — foreign writer.
    BadMagic,
    /// Version mismatch — foreign or generation-skewed writer.
    BadVersion,
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
// on the calling thread; the `MappedSeg` itself is never shared (`Send`
// for ownership transfer across threads; `!Sync` is intentional).
unsafe impl Send for MappedSeg {}

/// Open + mmap the current segment. A fresh mapping every call picks up
/// producer recreates (epoch or size changes) with no stale fd.
///
/// TOCTOU note: `fstat` runs before AND after `mmap`; a truncate inside
/// the remaining volatile-read window would `SIGBUS` on access. The
/// producer only ever unlink+recreates (never truncates live), and the
/// double-`fstat` narrows the race to a best-effort window — a documented
/// crash assumption shared with the previous generation, not a new risk.
fn map_segment(name: &CString, write: bool) -> Result<MappedSeg, ReadSkip> {
    let oflag = (if write { libc::O_RDWR } else { libc::O_RDONLY }) | libc::O_CLOEXEC;
    // Mode is ignored without O_CREAT; kept explicit for readability.
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
        return Err(ReadSkip::Absent);
    }
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        unsafe { libc::close(fd) };
        return Err(ReadSkip::Absent);
    }
    // `st_size` is signed; compare in-domain before casting (a negative
    // would wrap to a huge usize and only fail the equality anyway, but
    // be explicit).
    if stat.st_size != HEADER_SIZE as libc::off_t {
        unsafe { libc::close(fd) };
        return Err(ReadSkip::WrongSize);
    }
    let map_len = HEADER_SIZE;
    let prot = libc::PROT_READ | (if write { libc::PROT_WRITE } else { 0 });
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), map_len, prot, libc::MAP_SHARED, fd, 0) };
    if ptr == libc::MAP_FAILED || ptr.is_null() {
        unsafe { libc::close(fd) };
        return Err(ReadSkip::Absent);
    }
    // Re-check size after mapping: a recreate between the first fstat and
    // the mmap would otherwise leave a mapping over a replaced object.
    let mut stat2: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat2) } != 0 {
        unsafe {
            libc::munmap(ptr, map_len);
            libc::close(fd);
        }
        return Err(ReadSkip::Absent);
    }
    if stat2.st_size != HEADER_SIZE as libc::off_t {
        unsafe {
            libc::munmap(ptr, map_len);
            libc::close(fd);
        }
        return Err(ReadSkip::WrongSize);
    }
    Ok(MappedSeg {
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

    /// Seqlock-read of the v4 control header. `Ok` snapshot or `Err`
    /// [`ReadSkip`] reason — no logging in the loop beyond the mapper's
    /// throttled absent line and the exhaustion line below, so the caller
    /// aggregates counters for its heartbeat. Callers fall back to plain
    /// (color-only) views on every variant. The physical copy spans the
    /// whole mapping (including the pin words, which are discarded, never
    /// interpreted); the logical snapshot excludes `pinned[]` — those words
    /// live outside the seqlock (see [`Self::pin_slot`]). Bounded:
    /// [`READ_RETRIES`] tries, then [`ReadSkip::Torn`]. No delay between
    /// tries: the writer holds odd for one fill (nanoseconds), so a miss
    /// is either instantly retryable or a dead peer — sleeping would only
    /// add Submit-path latency.
    pub fn read_control(&self) -> Result<DepthControl, ReadSkip> {
        let map = map_segment(&self.name, false)?;
        let (ptr, map_len) = (map.ptr, map.len);
        debug_assert_eq!(map_len, HEADER_SIZE);
        for _ in 0..READ_RETRIES {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                // Writer inside (nanoseconds): pause-hint, not a sleep.
                std::hint::spin_loop();
                continue;
            }
            if unsafe { read_u32(ptr, 0) } != SHM_MAGIC {
                // Unstable seq means a concurrent publish tore these bytes
                // mid-read: retry (buckets as Torn on exhaustion), don't
                // mis-bucket a torn foreign-looking word as BadMagic.
                if unsafe { read_seq(ptr) } != seq1 {
                    continue;
                }
                return Err(ReadSkip::BadMagic);
            }
            if unsafe { read_u32(ptr, 4) } != SHM_VERSION {
                if unsafe { read_seq(ptr) } != seq1 {
                    continue;
                }
                return Err(ReadSkip::BadVersion);
            }
            // Copy under seq1; the closing seq check covers torn reads.
            let h = unsafe { read_header_v4(ptr) };
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            if h.magic != SHM_MAGIC {
                return Err(ReadSkip::BadMagic);
            }
            if h.version != SHM_VERSION {
                return Err(ReadSkip::BadVersion);
            }
            return Ok(DepthControl {
                epoch: h.epoch,
                w: h.w,
                h: h.h,
                format: h.format,
                generation: h.generation,
                time_ns: h.time_mono_ns,
                mem_fds: h.mem_fds,
                fence_fds: h.fence_fds,
                near_z: h.near_z,
                far_z: h.far_z,
                seg_len: map_len,
                alloc_bytes: h.alloc_bytes,
            });
        }
        // Exhausted: caller-invisible (no snapshot to bucket), so throttled
        // debug here like the absent path — never per-frame spam.
        static LAST_TORN_LOG_NS: AtomicU64 = AtomicU64::new(0);
        let now = now_mono_ns();
        if now.saturating_sub(LAST_TORN_LOG_NS.load(Ordering::Relaxed)) > 5_000_000_000 {
            LAST_TORN_LOG_NS.store(now, Ordering::Relaxed);
            debug!("depth feed: seqlock retries exhausted (torn)");
        }
        Err(ReadSkip::Torn)
    }

    fn pin_addr(map: &MappedSeg, slot: usize) -> Option<*mut AtomicU32> {
        if slot >= SLOTS || map.len != HEADER_SIZE {
            return None;
        }
        // Fail-safe to plain (not panic) on misalignment: `mmap` is page
        // aligned so `+80/+84` are always 4-aligned in practice; a hostile
        // mapping must never panic the app thread.
        // `AtomicU32` and `u32` are layout-identical; the cast only changes
        // how the word is accessed (atomic vs plain).
        // NOTE: `%` instead of `is_multiple_of` — workspace MSRV is 1.85.
        #[allow(clippy::manual_is_multiple_of)]
        if (map.ptr as usize + PINNED_OFF + slot * 4) % std::mem::align_of::<AtomicU32>() != 0 {
            return None;
        }
        Some(unsafe {
            (map.ptr as *mut u8)
                .add(PINNED_OFF + slot * 4)
                .cast::<AtomicU32>()
        })
    }

    /// Pin `slot` before polling/sampling it (`Release` store of 1).
    /// Pass `slot` from the generation just read (seqlock → pin → poll →
    /// sample; memory reimport on epoch change precedes the pin). At most
    /// one slot may be pinned at a time — unenforced here; holding both
    /// stalls the producer. The pin keeps the slot's fence FD stable: the
    /// producer closes and replaces fence FDs only on unpinned reuse.
    /// Returns false when the segment is absent or `slot >= 2`.
    pub fn pin_slot(&self, slot: usize) -> bool {
        let Ok(map) = map_segment(&self.name, true) else {
            return false;
        };
        let Some(addr) = Self::pin_addr(&map, slot) else {
            return false;
        };
        // Raw-pointer atomic op (no `&AtomicU32` reborrow): the word is
        // concurrently readable by the producer outside the Rust aliasing
        // model, so no shared reference is formed.
        unsafe {
            (*addr).store(1, Ordering::Release);
        }
        true
    }

    /// Unpin `slot` after the sampler's own fence confirms completion
    /// (not after submit). Returns false when absent or out of range.
    pub fn unpin_slot(&self, slot: usize) -> bool {
        let Ok(map) = map_segment(&self.name, true) else {
            return false;
        };
        let Some(addr) = Self::pin_addr(&map, slot) else {
            return false;
        };
        unsafe {
            (*addr).store(0, Ordering::Release);
        }
        true
    }

    /// Single `Acquire` load of a pin word. `None` when absent/out of range.
    /// Diagnostic/foreground use (pin-state asserts, future heartbeat
    /// inspection); the hot path pins/unpins without reading back.
    #[allow(dead_code)]
    pub fn load_pin(&self, slot: usize) -> Option<u32> {
        let Ok(map) = map_segment(&self.name, false) else {
            return None;
        };
        let addr = Self::pin_addr(&map, slot)?;
        Some(unsafe { (*addr).load(Ordering::Acquire) })
    }
}

/// # Safety
///
/// `base` must be a live `mmap`ed 96-byte header for the call duration.
/// `mmap` is page-aligned, so offset 8 is 4-aligned; same-machine SHM is
/// native-endian.
unsafe fn read_seq(base: *mut libc::c_void) -> u32 {
    unsafe { (base as *const u8).add(8).cast::<u32>().read_volatile() }
}

/// # Safety
///
/// Same as [`read_seq`]; `off` must be 4-aligned and in-bounds.
unsafe fn read_u32(base: *mut libc::c_void, off: usize) -> u32 {
    // Seqlock-idiom volatile read: inhibits compiler elision/reorder of
    // the seq→body→seq triple. Formally this relies on platform
    // single-copy atomicity of aligned u32 plus the seqlock retry — not
    // on volatile being a substitute for atomics.
    debug_assert_eq!(off % 4, 0);
    unsafe { (base as *const u8).add(off).cast::<u32>().read_volatile() }
}

/// # Safety
///
/// Same as [`read_seq`]; the whole 96-byte object must be mapped.
/// The copy spans the pin words; callers discard them (see
/// [`DepthFeed::read_control`]).
unsafe fn read_header_v4(base: *mut libc::c_void) -> XdShmHeaderV4 {
    // Same seqlock idiom as above; torn copies are detected by the
    // closing seq check, never trusted directly.
    debug_assert_eq!(base as usize % std::mem::align_of::<XdShmHeaderV4>(), 0);
    unsafe { (base as *const XdShmHeaderV4).read_volatile() }
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
/// This is the ONLY submission gate: there is no absolute-age "stale"
/// check. Same-generation success implies fresh (the window sits strictly
/// inside any sane liveness bound); a stalled producer shows up as a
/// frozen `generation` across frames, not a clock comparison.
/// Callers may additionally reject `time_ns == 0` (never-published
/// sentinel): a failed receipt clock (0) coinciding with a zero stamp
/// reads as age 0 and would otherwise pass.
pub fn is_same_generation(age_ns: i64, frame_period_ns: i64, margin_ns: i64) -> bool {
    debug_assert!(margin_ns >= 0 && frame_period_ns >= 0);
    age_ns >= margin_ns.saturating_neg() && age_ns < frame_period_ns.saturating_add(margin_ns)
}

/// Running signed frame-vs-depth age stats for one heartbeat window.
/// Feed one [`depth_age_ns`] sample per successful read (misses have no
/// timestamp — count those via [`ReadSkip`]). Integer state only (no float,
/// no alloc); the heartbeat formats the summary. Reset with [`AgeStats::clear`]
/// after each heartbeat line.
#[derive(Debug, Default, Clone)]
pub struct AgeStats {
    count: u64,
    sum: i64,
    /// Most positive sample (stalest depth).
    max: i64,
    /// Most negative sample (furthest straddled present).
    min: i64,
    neg_count: u64,
}

impl AgeStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one signed age sample. Hot path: integer ops only.
    pub fn sample(&mut self, age_ns: i64) {
        if self.count == 0 {
            self.max = age_ns;
            self.min = age_ns;
        } else {
            self.max = self.max.max(age_ns);
            self.min = self.min.min(age_ns);
        }
        self.sum = self.sum.saturating_add(age_ns);
        self.count = self.count.saturating_add(1);
        if age_ns < 0 {
            self.neg_count = self.neg_count.saturating_add(1);
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Mean signed age in ns, or `None` when empty. Positive mean = receipt
    /// consistently lags present (phase info); near-zero = aligned.
    pub fn mean_ns(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(self.sum as f64 / self.count as f64)
    }

    /// Most positive sample, or `None` when empty.
    pub fn max_ns(&self) -> Option<i64> {
        if self.count == 0 {
            return None;
        }
        Some(self.max)
    }

    /// Most negative sample, or `None` when empty.
    pub fn min_ns(&self) -> Option<i64> {
        if self.count == 0 {
            return None;
        }
        Some(self.min)
    }

    /// Fraction of straddled-present (negative-age) samples in `[0, 1]`.
    /// Validates the Submit-before-Present assumption empirically.
    pub fn neg_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.neg_count as f64 / self.count as f64
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

static WARNED_MASTER: AtomicBool = AtomicBool::new(false);
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

/// `CLOCK_MONOTONIC` in ns. Same clock the producer stamps `time_mono_ns`
/// with, same machine. Returns 0 on `clock_gettime` failure; [`depth_age_ns`]
/// maps 0 to a large negative age that [`is_same_generation`] rejects, so
/// the failure direction is fail-safe to plain frames.
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

/// Poll outcome for [`poll_fence_fd`]. `Copy` enum — the heartbeat buckets
/// it with counter increments, no formatting on the frame path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FencePoll {
    /// Fence signaled: the producer's copy for this generation completed.
    Ready,
    /// Still pending after `timeout_ms`: skip (plain), keep the pin only if
    /// a later poll is planned, else unpin. Bounded cost by construction.
    Timeout,
    /// FD invalid (`<= 2`), closed under us (segment vanished), or poll
    /// errored: the snapshot is dead — re-read, never wait on it.
    Dead,
}

/// Poll one published fence sync_file for producer-copy completion.
///
/// `fd` is the pinned slot's current fence number, used directly (poll
/// takes no ownership; same-process borrowed numbers). Returns when
/// signaled or after `timeout_ms`, whichever comes first — never blocks
/// indefinitely (`timeout_ms` is clamped at zero: negative would mean
/// infinite per `poll(2)`). Call only while holding the slot's pin: the
/// producer closes and replaces fence FDs on unpinned reuse, so an
/// unpinned poll can observe a closed or recycled number (reported as
/// [`FencePoll::Dead`] via `POLLNVAL`/error, fail-safe to plain).
pub fn poll_fence_fd(fd: i32, timeout_ms: libc::c_int) -> FencePoll {
    if fd <= 2 {
        return FencePoll::Dead;
    }
    let timeout_ms = timeout_ms.max(0);
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: pfd points to one valid local pollfd; timeout is bounded.
    let r = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if r > 0 {
        // IN = signaled (HUP may accompany it on a consumed file — still
        // signaled). Anything else with r > 0 (NVAL = closed under us, ERR,
        // HUP-alone) means the snapshot is unusable: never wait on it.
        if pfd.revents & libc::POLLNVAL != 0 {
            FencePoll::Dead
        } else if pfd.revents & libc::POLLIN != 0 {
            FencePoll::Ready
        } else {
            FencePoll::Dead
        }
    } else if r == 0 {
        FencePoll::Timeout
    } else {
        // -1: EINTR (a signal stole the wait) or EBADF (raced close) —
        // either way the snapshot is unusable; re-read next frame.
        FencePoll::Dead
    }
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

    fn make_seg(
        name: &str,
        w: u32,
        h: u32,
        generation: u64,
        tweak: impl FnOnce(&mut XdShmHeaderV4),
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
        let mut hdr = XdShmHeaderV4 {
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
            fence_fds: [21, 22],
            alloc_bytes: 8 * 2 * 4,
            near_z: 0.00742,
            far_z: 10000.0,
            reserved_pad: 0,
            pinned: [0, 0],
            reserved: [0, 0],
        };
        tweak(&mut hdr);
        unsafe {
            (ptr as *mut XdShmHeaderV4).write_unaligned(hdr);
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
        // Byte-identical with reprojex/src/shm.h (v4, 96B).
        assert_eq!(std::mem::size_of::<XdShmHeaderV4>(), 96);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, magic), 0);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, version), 4);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, seq), 8);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, w), 12);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, h), 16);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, format), 20);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, epoch), 24);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, reserved0), 28);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, generation), 32);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, time_mono_ns), 40);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, mem_fds), 48);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, fence_fds), 56);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, alloc_bytes), 64);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, near_z), 68);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, far_z), 72);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, reserved_pad), 76);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, pinned), PINNED_OFF);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, pinned) + 4, 84);
        assert_eq!(std::mem::offset_of!(XdShmHeaderV4, reserved), 88);
    }

    #[test]
    fn wire_constants_pinned() {
        // A drifted rendezvous constant is a cross-process break; pin values.
        assert_eq!(READ_RETRIES, 4);
        assert_eq!(SHM_NAME, "/xplane-depth-v4");
        assert_eq!(SHM_MAGIC, 0x5850_4450);
        assert_eq!(SHM_VERSION, 4);
        assert_eq!(HEADER_SIZE, 96);
        assert_eq!(SLOTS, 2);
        assert_eq!(PINNED_OFF, 80);
        assert_eq!(FORMAT_D32F, 126);
        assert_eq!(FORMAT_D32S8, 130);
        assert_eq!((MIN_DEPTH, MAX_DEPTH), (0.0, 1.0));
    }

    #[test]
    fn viable_snapshot_roundtrip() {
        let name = format!("/xrizer-depth-test-{}-1", std::process::id());
        let _seg = make_seg(&name, 8, 2, 42, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("should hit");
        assert_eq!((ctl.w, ctl.h, ctl.generation), (8, 2, 42), "dims+gen");
        assert_eq!(ctl.epoch, 7, "epoch");
        assert_eq!(ctl.mem_fds, [11, 12], "mem fds");
        assert_eq!(ctl.fence_fds, [21, 22], "fence fds");
        assert_eq!(ctl.alloc_bytes, 8 * 2 * 4, "alloc");
        assert!((ctl.near_z - 0.00742).abs() < f32::EPSILON, "near");
        assert!((ctl.far_z - 10000.0).abs() < f32::EPSILON, "far");
        assert!(ctl.is_viable(), "viable");
        assert_eq!(ctl.eye_w(), 4, "eye_w = w/2");
        assert_eq!(ctl.slot(), 0, "42 % 2"); // 42 % 2
        // Fresh snapshot: age near zero lands inside the same-generation
        // window (90Hz period + 2ms margin).
        let age = ctl.age_ns(now_mono_ns());
        assert!(
            is_same_generation(age, 11_111_111, 2_000_000),
            "fresh age {age} in window"
        );
        assert!(ctl.alloc_covers(4), "alloc covers bpp=4");
        assert!(!ctl.alloc_covers(4 * 1000), "alloc rejects huge want");
    }

    #[test]
    fn reserved_nonzero_still_viable_and_alloc_zero_tolerated() {
        // Forward-compat: reserved words are never checked.
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-rsv");
        let _seg = make_seg(&name, 8, 2, 3, |h| {
            h.reserved0 = 0xdead_beef;
            h.reserved_pad = 1;
            h.reserved = [7, 8];
            h.alloc_bytes = 0;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable(), "reserved nonzero must not gate");
        assert!(
            ctl.alloc_covers(4),
            "alloc==0 tolerates any want as unknown"
        );
        assert!(
            ctl.alloc_covers(u64::MAX),
            "alloc==0 tolerates even huge want"
        );
    }

    #[test]
    fn viability_excludes_alloc_and_near_far() {
        // Caller must AND all three: viable snapshot with bad clip and
        // undersized alloc is still is_viable()==true.
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-split");
        let _seg = make_seg(&name, 8, 2, 3, |h| {
            h.near_z = 0.0;
            h.far_z = 0.0;
            h.alloc_bytes = 1;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable(), "alloc/clip must not gate viability");
        assert_eq!(
            validate_near_far(ctl.near_z, ctl.far_z),
            None,
            "clip invalid"
        );
        assert!(!ctl.alloc_covers(4), "alloc undersized");
    }

    #[test]
    fn slot_mapping_is_gen_mod_two() {
        assert_eq!(DepthControl::slot_for_generation(0), 0);
        assert_eq!(DepthControl::slot_for_generation(1), 1);
        assert_eq!(DepthControl::slot_for_generation(2), 0);
        assert_eq!(DepthControl::slot_for_generation(43), 1);
        assert_eq!(DepthControl::slot_for_generation(u64::MAX), 1);
    }

    #[test]
    fn dims_boundaries_and_eye_w() {
        let pid = std::process::id();
        // Max publishable dims are viable.
        let name = format!("/xrizer-depth-test-{pid}-maxdims");
        let _seg = make_seg(&name, 16384, 16384, 3, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable(), "16384x16384 viable");
        // want_bytes must not wrap at max dims (saturating).
        assert_eq!(ctl.want_bytes(4), 16384u64 * 16384 * 4);
        // Zero/huge on either axis rejected.
        for (tag, w, h) in [
            ("huge-h", 8u32, 16385u32),
            ("zero-h", 8u32, 0u32),
            ("zero-w", 0u32, 8u32),
        ] {
            let name = format!("/xrizer-depth-test-{pid}-{tag}");
            let _seg = make_seg(&name, w, h, 3, |_| {});
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("readable");
            assert!(!ctl.is_viable(), "{tag} must not arm");
        }
        // Odd h is fine (only w must be even for the SBS split).
        let name = format!("/xrizer-depth-test-{pid}-oddh");
        let _seg = make_seg(&name, 8, 3, 3, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable(), "odd h viable");
        // Pure-math eye_w truncation (odd w never reaches here via SHM,
        // but the helper truncates rather than rounds).
        let mk = |w: u32| DepthControl {
            epoch: 1,
            w,
            h: 2,
            format: FORMAT_D32F,
            generation: 1,
            time_ns: 1,
            mem_fds: [11, 12],
            fence_fds: [21, 22],
            near_z: 0.01,
            far_z: 200.0,
            seg_len: HEADER_SIZE,
            alloc_bytes: 0,
        };
        assert_eq!(mk(8).eye_w(), 4);
        assert_eq!(mk(7).eye_w(), 3);
        // Hand-built seg_len gate (map path already enforces ==96).
        let mut bad_len = mk(8);
        bad_len.seg_len = 95;
        assert!(!bad_len.is_viable(), "seg_len != 96 must not arm");
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
        assert_eq!(
            DepthFeed::open_for_test(&name).read_control().unwrap_err(),
            ReadSkip::WrongSize,
            "v2 size must miss"
        );
        // 96-byte segment stamped version 2: wrong version.
        let name2 = format!("/xrizer-depth-test-{pid}-v2b");
        let _seg2 = make_seg(&name2, 8, 2, 1, |h| {
            h.version = 2;
        });
        assert_eq!(
            DepthFeed::open_for_test(&name2).read_control().unwrap_err(),
            ReadSkip::BadVersion,
            "version 2 must miss"
        );
        // Future versions are equally foreign.
        for v in [0u32, 3, 5, u32::MAX] {
            let name = format!("/xrizer-depth-test-{pid}-v{v}");
            let _seg = make_seg(&name, 8, 2, 1, |h| {
                h.version = v;
            });
            assert_eq!(
                DepthFeed::open_for_test(&name).read_control().unwrap_err(),
                ReadSkip::BadVersion,
                "version {v} must miss"
            );
        }
    }

    #[test]
    fn bad_magic_and_size_rejected() {
        let pid = std::process::id();
        let name = format!("/xrizer-depth-test-{pid}-mag");
        let seg = make_seg(&name, 8, 2, 1, |_| {});
        unsafe {
            (seg.ptr as *mut u32).write_volatile(0xdead_beef);
            (seg.ptr as *mut u32).add(2).write_volatile(2);
        }
        assert_eq!(
            DepthFeed::open_for_test(&name).read_control().unwrap_err(),
            ReadSkip::BadMagic,
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
        assert_eq!(
            DepthFeed::open_for_test(&name2).read_control().unwrap_err(),
            ReadSkip::WrongSize,
            "wrong size must miss"
        );
        // Truncated segments (mid-recreate races): same gate. (A 0-byte
        // segment needs no mapping — fstat alone rejects it — so the
        // smallest mappable case here is 1 byte.)
        for (tag, len) in [("s64", 64usize), ("s95", 95usize), ("s1", 1usize)] {
            let name = format!("/xrizer-depth-test-{pid}-{tag}");
            let _seg = make_raw_seg(&name, len, |ptr| unsafe {
                if len >= 12 {
                    std::ptr::write_bytes(ptr as *mut u8, 0, len);
                    (ptr as *mut u32).write_volatile(SHM_MAGIC);
                    (ptr as *mut u32).add(1).write_volatile(SHM_VERSION);
                    (ptr as *mut u32).add(2).write_volatile(2);
                }
            });
            assert_eq!(
                DepthFeed::open_for_test(&name).read_control().unwrap_err(),
                ReadSkip::WrongSize,
                "{tag} ({len}B) must miss"
            );
        }
    }

    #[test]
    fn sentinel_and_low_fds_rejected() {
        let pid = std::process::id();
        let cases: [(i32, i32); 5] = [(-1, 21), (0, 21), (1, 21), (2, 21), (11, 2)];
        for (i, (mem0, fence0)) in cases.iter().enumerate() {
            let name = format!("/xrizer-depth-test-{pid}-fd{i}");
            let _seg = make_seg(&name, 8, 2, 3, |h| {
                h.mem_fds[0] = *mem0;
                h.fence_fds[0] = *fence0;
            });
            let ctl = DepthFeed::open_for_test(&name)
                .read_control()
                .expect("header readable");
            assert!(
                !ctl.fds_valid() && !ctl.is_viable(),
                "fds ({mem0}, {fence0}) must not arm"
            );
        }
        // Second lane is checked too, plus the all-sentinel pre-publish.
        let name = format!("/xrizer-depth-test-{pid}-fd-slot1");
        let _seg = make_seg(&name, 8, 2, 3, |h| {
            h.mem_fds[1] = -1;
            h.fence_fds[1] = -1;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("header readable");
        assert!(
            !ctl.fds_valid() && !ctl.is_viable(),
            "slot-1 sentinel must not arm"
        );
        let name = format!("/xrizer-depth-test-{pid}-fd-allneg");
        let _seg = make_seg(&name, 8, 2, 3, |h| {
            h.mem_fds = [-1, -1];
            h.fence_fds = [-1, -1];
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("header readable");
        assert!(!ctl.fds_valid() && !ctl.is_viable(), "all -1 must not arm");
    }

    #[test]
    fn pre_publish_and_dims_rejected() {
        let pid = std::process::id();
        // epoch 0 with live generation: never published (zero-init mapping).
        let name = format!("/xrizer-depth-test-{pid}-pre-epoch");
        let _seg = make_seg(&name, 8, 2, 42, |h| {
            h.epoch = 0;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "epoch 0 must not arm");
        // generation 0 with live epoch: nothing completed yet.
        let name = format!("/xrizer-depth-test-{pid}-pre-gen");
        let _seg = make_seg(&name, 8, 2, 0, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "generation 0 must not arm");
        // Bad dims / format.
        let name = format!("/xrizer-depth-test-{pid}-oddw");
        let _seg = make_seg(&name, 7, 2, 3, |_| {});
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(!ctl.is_viable(), "odd width must not arm");
        type Tweak = (&'static str, fn(&mut XdShmHeaderV4));
        let tweaks: [Tweak; 5] = [
            ("zerow", |h: &mut XdShmHeaderV4| h.w = 0),
            ("huge", |h: &mut XdShmHeaderV4| h.w = 16385),
            ("fmt", |h: &mut XdShmHeaderV4| h.format = 96),
            ("gen0", |h: &mut XdShmHeaderV4| h.generation = 0),
            ("epoch0", |h: &mut XdShmHeaderV4| h.epoch = 0),
        ];
        for (tag, tweak) in tweaks {
            let name = format!("/xrizer-depth-test-{pid}-{tag}");
            let _seg = make_seg(&name, 8, 2, 3, tweak);
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
        let _seg = make_seg(&name, 8, 2, 3, |h| {
            h.format = FORMAT_D32S8;
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert_eq!(ctl.format, FORMAT_D32S8);
        assert!(ctl.is_viable(), "D32S8 fallback must arm");
    }

    #[test]
    fn age_gate_is_symmetric_frame_vs_depth() {
        // No absolute "stale" gate exists: submission is gated purely on the
        // signed frame-vs-depth age. Old AND far-future headers both read
        // fine (viable) — the tight same-generation window rejects them.
        let mk = |time_ns: u64| DepthControl {
            epoch: 1,
            w: 8,
            h: 2,
            format: FORMAT_D32F,
            generation: 1,
            time_ns,
            mem_fds: [11, 12],
            fence_fds: [21, 22],
            near_z: 0.00742,
            far_z: 10000.0,
            seg_len: HEADER_SIZE,
            alloc_bytes: 64,
        };
        // 90 Hz frame: 11_111_111ns period, 2ms margin.
        let receipt = 10_000_000_000u64;
        // 500ms-old depth: viable snapshot, rejected association.
        let old = mk(receipt - 500_000_000);
        assert!(old.is_viable());
        let age = old.age_ns(receipt);
        assert_eq!(age, 500_000_000);
        assert!(!is_same_generation(age, 11_111_111, 2_000_000));
        // Far-future depth (clock breakage): symmetric rejection.
        let fut = mk(receipt + 5_000_000_000);
        assert!(fut.is_viable(), "future still reads viable");
        let age = fut.age_ns(receipt);
        assert_eq!(age, -5_000_000_000);
        assert!(!is_same_generation(age, 11_111_111, 2_000_000));
        // Small-negative (straddled present, freshest case): accepted.
        let fresh = mk(receipt + 1_000_000);
        assert!(is_same_generation(
            fresh.age_ns(receipt),
            11_111_111,
            2_000_000
        ));
        // Zero stamp: viable snapshot (time is not a viability input),
        // but the huge age misses every window.
        let zero = mk(0);
        assert!(zero.is_viable(), "time is not viability-gated");
        assert!(!is_same_generation(
            zero.age_ns(receipt),
            11_111_111,
            2_000_000
        ));
    }

    #[test]
    fn old_and_future_headers_read_viable_with_signed_ages() {
        let pid = std::process::id();
        // Headers far outside any association window still snapshot fine:
        // staleness is decided by the caller from the signed age, not here.
        let name = format!("/xrizer-depth-test-{pid}-oldhdr");
        let now_setup = now_mono_ns();
        let _seg = make_seg(&name, 8, 2, 7, |h| {
            h.time_mono_ns = now_setup.saturating_sub(500_000_000);
        });
        let ctl = DepthFeed::open_for_test(&name)
            .read_control()
            .expect("readable");
        assert!(ctl.is_viable(), "old header still viable snapshot");
        let age = ctl.age_ns(now_mono_ns());
        // One-sided: post-read time can only be later, so slow CI only
        // grows the age — an upper bound would flake.
        assert!(age >= 499_000_000, "age >= ~500ms, got {age}");
        assert!(!is_same_generation(age, 11_111_111, 2_000_000));
        let name2 = format!("/xrizer-depth-test-{pid}-futhdr");
        let _seg2 = make_seg(&name2, 8, 2, 8, |h| {
            h.time_mono_ns = now_mono_ns().saturating_add(5_000_000_000);
        });
        let ctl2 = DepthFeed::open_for_test(&name2)
            .read_control()
            .expect("readable");
        assert!(ctl2.is_viable(), "future header still viable snapshot");
        let age2 = ctl2.age_ns(now_mono_ns());
        assert!(age2 < -4_000_000_000, "age hugely negative, got {age2}");
        assert!(!is_same_generation(age2, 11_111_111, 2_000_000));
    }

    #[test]
    fn pin_roundtrip_outside_seqlock() {
        let name = format!("/xrizer-depth-test-{}-pin", std::process::id());
        let seg = make_seg(&name, 8, 2, 42, |_| {});
        // Independent raw view through the test's own mapping at hardcoded
        // +80/+84 (not via PINNED_OFF): proves the ABI offset, not just
        // self-consistency. Atomic loads — a plain read would race.
        let raw_pin = |slot: usize| unsafe {
            (seg.ptr as *const u8)
                .add(80 + slot * 4)
                .cast::<AtomicU32>()
                .as_ref()
                .expect("test mapping live")
                .load(Ordering::Acquire)
        };
        let feed = DepthFeed::open_for_test(&name);
        // Fresh segment boots unpinned; snapshot reads fine alongside pins.
        assert_eq!(feed.load_pin(0), Some(0));
        assert_eq!(feed.load_pin(1), Some(0));
        assert_eq!((raw_pin(0), raw_pin(1)), (0, 0), "raw +80/+84 boot at 0");
        assert!(feed.read_control().is_ok());
        // Pin slot 0: raw +80 flips to 1 while +84 stays 0.
        assert!(feed.pin_slot(0));
        assert_eq!(feed.load_pin(0), Some(1));
        assert_eq!(feed.load_pin(1), Some(0));
        assert_eq!((raw_pin(0), raw_pin(1)), (1, 0), "raw +80==1,+84==0");
        // Snapshot still clean while pinned (pins are outside the seqlock).
        assert!(feed.read_control().is_ok());
        // Pin slot 1 as well (API permits; caller must hold at most one —
        // both set proves the words are independent).
        assert!(feed.pin_slot(1));
        assert_eq!((feed.load_pin(0), feed.load_pin(1)), (Some(1), Some(1)));
        assert_eq!((raw_pin(0), raw_pin(1)), (1, 1), "raw +80==1,+84==1");
        assert!(feed.unpin_slot(0));
        assert!(feed.unpin_slot(1));
        assert_eq!(feed.load_pin(0), Some(0));
        assert_eq!(feed.load_pin(1), Some(0));
        // Out-of-range slots never touch the mapping.
        assert!(!feed.pin_slot(2));
        assert!(!feed.unpin_slot(2));
        assert_eq!(feed.load_pin(2), None);
        assert_eq!((raw_pin(0), raw_pin(1)), (0, 0), "raw words rest at 0");
    }

    #[test]
    fn stuck_odd_seqlock_returns_torn() {
        let name = format!("/xrizer-depth-test-{}-odd", std::process::id());
        let seg = make_seg(&name, 8, 2, 1, |_| {});
        unsafe {
            (seg.ptr as *mut u32).add(2).write_volatile(3);
        }
        let feed = DepthFeed::open_for_test(&name);
        assert_eq!(feed.read_control().unwrap_err(), ReadSkip::Torn);
    }

    #[test]
    fn absent_returns_absent() {
        let feed = DepthFeed::open_for_test("/xrizer-depth-test-definitely-absent");
        assert_eq!(feed.read_control().unwrap_err(), ReadSkip::Absent);
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
        // Saturation at the extremes (untrusted u64 inputs).
        assert_eq!(depth_age_ns(u64::MAX, 0), i64::MAX);
        assert_eq!(depth_age_ns(0, u64::MAX), -i64::MAX);
        // Same-generation window: last period plus small negative margin.
        assert!(is_same_generation(5_000_000, 11_111_111, 2_000_000));
        assert!(is_same_generation(-1_000_000, 11_111_111, 2_000_000));
        assert!(!is_same_generation(-3_000_000, 11_111_111, 2_000_000));
        assert!(!is_same_generation(15_000_000, 11_111_111, 2_000_000));
        // Boundaries: -margin inclusive, period+margin exclusive.
        assert!(is_same_generation(-2_000_000, 11_111_111, 2_000_000));
        assert!(!is_same_generation(-2_000_001, 11_111_111, 2_000_000));
        assert!(!is_same_generation(13_111_111, 11_111_111, 2_000_000));
        assert!(is_same_generation(13_111_110, 11_111_111, 2_000_000));
        // Overflow-safe at the extremes.
        assert!(is_same_generation(0, i64::MAX, i64::MAX));
    }

    #[test]
    fn env_parsing_strict() {
        let _lock = env_guard();
        let save = |key: &str| std::env::var(key).ok();
        let orig_master = save("XRIZER_DEPTH_SHM");
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

        // Master edge rows: empty and out-of-vocabulary default ON.
        set_env_for_tests("XRIZER_DEPTH_SHM", "");
        assert!(depth_master_enabled(), "empty defaults ON");
        set_env_for_tests("XRIZER_DEPTH_SHM", "2");
        assert!(depth_master_enabled(), "2 defaults ON");

        restore("XRIZER_DEPTH_SHM", orig_master);
    }

    #[test]
    fn open_respects_kill_switch() {
        // NOTE: env-mutating; serialized through the same guard. Any future
        // test calling open() must hold env_guard() too.
        let _lock = env_guard();
        let save = std::env::var("XRIZER_DEPTH_SHM").ok();
        set_env_for_tests("XRIZER_DEPTH_SHM", "0");
        assert!(DepthFeed::open().is_none(), "kill-switch off");
        set_env_for_tests("XRIZER_DEPTH_SHM", "bogus");
        assert!(DepthFeed::open().is_some(), "garbage fail-open on");
        del_env_for_tests("XRIZER_DEPTH_SHM");
        assert!(DepthFeed::open().is_some(), "default on");
        match save {
            Some(v) => set_env_for_tests("XRIZER_DEPTH_SHM", &v),
            None => del_env_for_tests("XRIZER_DEPTH_SHM"),
        }
    }

    #[test]
    fn read_skip_variants_are_distinct_and_copy() {
        // Heartbeat bucketing relies on cheap discriminant compares.
        fn bucket(skip: ReadSkip) -> &'static str {
            match skip {
                ReadSkip::Absent => "absent",
                ReadSkip::WrongSize => "size",
                ReadSkip::Torn => "torn",
                ReadSkip::BadMagic => "magic",
                ReadSkip::BadVersion => "version",
            }
        }
        assert_eq!(bucket(ReadSkip::Absent), "absent");
        assert_eq!(bucket(ReadSkip::Torn), "torn");
        let a = ReadSkip::WrongSize;
        let b = a;
        assert_eq!(a, b, "Copy semantics for counter keys");
        assert_ne!(a, ReadSkip::BadMagic);
    }

    #[test]
    fn age_stats_empty_mean_and_rates() {
        let s = AgeStats::new();
        assert_eq!(s.count(), 0);
        assert_eq!(s.mean_ns(), None);
        assert_eq!(s.max_ns(), None);
        assert_eq!(s.min_ns(), None);
        assert_eq!(s.neg_rate(), 0.0);
    }

    #[test]
    fn age_stats_tracks_mean_max_min_and_neg_rate() {
        let mut s = AgeStats::new();
        for age in [5_000_000i64, -1_000_000, 9_000_000, -2_000_000, 4_000_000] {
            s.sample(age);
        }
        assert_eq!(s.count(), 5);
        // (5 - 1 + 9 - 2 + 4)ms / 5 = 3ms.
        assert!((s.mean_ns().expect("nonempty") - 3_000_000.0).abs() < 1.0);
        assert_eq!(s.max_ns(), Some(9_000_000), "stalest");
        assert_eq!(s.min_ns(), Some(-2_000_000), "furthest straddle");
        assert!((s.neg_rate() - 0.4).abs() < f64::EPSILON, "2/5 negative");
    }

    #[test]
    fn age_stats_single_sample_and_clear() {
        let mut s = AgeStats::new();
        s.sample(-500);
        assert_eq!(
            (s.mean_ns(), s.max_ns(), s.min_ns()),
            (Some(-500.0), Some(-500), Some(-500))
        );
        assert_eq!(s.neg_rate(), 1.0);
        s.clear();
        assert_eq!(s.count(), 0);
        assert_eq!(s.mean_ns(), None);
        s.sample(7);
        assert_eq!(s.count(), 1);
        assert_eq!(s.mean_ns(), Some(7.0));
        assert_eq!(s.max_ns(), Some(7));
        assert_eq!(s.min_ns(), Some(7));
        assert_eq!(s.neg_rate(), 0.0);
    }

    /// Fence-poll outcomes against real (pipe-backed) pollable FDs. Pipes
    /// stand in for sync_file FDs: both signal `POLLIN`, time out empty,
    /// and report dead when closed — the only three behaviors the caller
    /// depends on.
    #[test]
    fn fence_poll_ready_timeout_dead() {
        assert_eq!(poll_fence_fd(-1, 0), FencePoll::Dead);
        assert_eq!(poll_fence_fd(0, 0), FencePoll::Dead);
        assert_eq!(poll_fence_fd(2, 0), FencePoll::Dead);
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let (r, w) = (fds[0], fds[1]);
        // Empty pipe, zero timeout: pending.
        assert_eq!(poll_fence_fd(r, 0), FencePoll::Timeout);
        // One byte readable: signaled.
        assert_eq!(unsafe { libc::write(w, [1u8].as_ptr().cast(), 1) }, 1);
        assert_eq!(poll_fence_fd(r, 10), FencePoll::Ready);
        // Drained pipe blocks again (bounded 1ms check, not zero, so a
        // slow CI scheduler can't flake the direction).
        let mut buf = [0u8; 1];
        assert_eq!(unsafe { libc::read(r, buf.as_mut_ptr().cast(), 1) }, 1);
        assert_eq!(poll_fence_fd(r, 1), FencePoll::Timeout);
        unsafe {
            libc::close(r);
            libc::close(w);
        }
        // Closed read end: dead, never waited on (timeout 0: the error
        // returns immediately, no block window for a reuse race).
        assert_eq!(poll_fence_fd(r, 0), FencePoll::Dead);
    }

    #[test]
    fn fence_poll_copy_and_heartbeat_key() {
        // Heartbeat bucketing relies on cheap discriminant compares.
        let a = FencePoll::Ready;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, FencePoll::Timeout);
        assert_ne!(a, FencePoll::Dead);
    }
}
