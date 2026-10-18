//! SHM depth handoff reader (X-Plane layer -> xrizer).
//!
//! Transport is a POSIX SHM latest-slot segment (`/xplane-depth-v1` by default,
//! overridable via `XP_DEPTH_SHM_NAME` for tests). Layout is little-endian,
//! all offsets fixed:
//!
//! ```text
//! +0  u32 magic       0x58504450 ('XPDP')
//! +4  u32 version     1
//! +8  u32 seq         seqlock: odd = writer inside
//! +12 u32 w, +16 u32 h
//! +20 i32 format      Vulkan format enum (126 = D32F)
//! +24 u32 eye_layout  1 = side-by-side (left half eye0, right half eye1)
//! +32 u64 present_no
//! +40 u64 time_mono_ns (CLOCK_MONOTONIC when copy completed)
//! +48 f32 min_depth, +52 f32 max_depth
//! +56 u32 reserved[2]
//! +64 f32[w*h] row-major pixels follow
//! ```
//!
//! No new crates: `libc` only. Absent/stale/mismatched SHM always yields
//! `None` so callers fall back to plain (color-only) views.

use log::{debug, info};
use std::ffi::CString;

pub const SHM_MAGIC: u32 = 0x5850_4450; // 'XPDP'
pub const SHM_VERSION: u32 = 1;
pub const EYE_SIDEBYSIDE: u32 = 1;
/// Vulkan `VK_FORMAT_D32_SFLOAT`.
pub const FORMAT_D32F: i32 = 126;
pub const HEADER_SIZE: usize = 64;

pub const DEFAULT_SHM_NAME: &str = "/xplane-depth-v1";
pub const GL_DEPTH_COMPONENT32F: u32 = 0x8CAC;

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

/// Owned copy of one SHM publish.
#[derive(Debug)]
pub struct DepthFrame {
    pub w: u32,
    pub h: u32,
    pub present_no: u64,
    pub time_ns: u64,
    pub pixels: Vec<f32>,
}

impl DepthFrame {
    pub fn eye_w(&self) -> u32 {
        self.w / 2
    }
}

/// Reader handle. Stateless across frames (open/mmap per read) so producer
/// `munmap/shm_unlink/recreate` cycles on frame-graph rebuilds are picked up
/// automatically with no stale fd.
pub struct DepthFeed {
    name: CString,
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
        let fd =
            unsafe { libc::shm_open(self.name.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC, 0o600) };
        if fd < 0 {
            debug!("depth feed: shm_open failed (absent)");
            return None;
        }
        struct Guard {
            fd: libc::c_int,
            ptr: *mut libc::c_void,
            len: usize,
        }
        impl Drop for Guard {
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

        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 {
            unsafe { libc::close(fd) };
            return None;
        }
        let map_len = stat.st_size as usize;
        if map_len < HEADER_SIZE {
            unsafe { libc::close(fd) };
            return None;
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return None;
        }
        let _guard = Guard {
            fd,
            ptr,
            len: map_len,
        };

        // Seqlock read: retry while odd or changed. Bounded; writer publishes
        // rarely (dump ticks only) so contention is essentially impossible.
        for _ in 0..4 {
            let seq1 = unsafe { read_seq(ptr) };
            if seq1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            // Copy header + pixels under seq1.
            let hdr = unsafe { read_header(ptr) };
            if !header_plausible(&hdr, map_len) {
                return None;
            }
            let npix = hdr.w as usize * hdr.h as usize;
            let mut pixels = vec![0f32; npix];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (ptr as *const u8).add(HEADER_SIZE) as *const f32,
                    pixels.as_mut_ptr(),
                    npix,
                );
            }
            let seq2 = unsafe { read_seq(ptr) };
            if seq1 != seq2 {
                continue;
            }
            // Accept only magic+version, float32, side-by-side.
            if hdr.magic != SHM_MAGIC || hdr.version != SHM_VERSION {
                return None;
            }
            if hdr.format != FORMAT_D32F || hdr.eye_layout != EYE_SIDEBYSIDE {
                return None;
            }
            // Staleness: future data or older than max age.
            if hdr.time_mono_ns > now_mono_ns {
                return None;
            }
            let max_age_ns = max_age_ms.saturating_mul(1_000_000);
            if now_mono_ns.saturating_sub(hdr.time_mono_ns) > max_age_ns {
                return None;
            }
            return Some(DepthFrame {
                w: hdr.w,
                h: hdr.h,
                present_no: hdr.present_no,
                time_ns: hdr.time_mono_ns,
                pixels,
            });
        }
        None
    }
}

unsafe fn read_seq(base: *mut libc::c_void) -> u32 {
    unsafe { (base as *const u8).add(8).cast::<u32>().read_volatile() }
}

unsafe fn read_header(base: *mut libc::c_void) -> XdShmHeader {
    // Field-by-field volatile read is overkill; the seqlock re-check covers
    // torn reads. Plain copy out of the mapping.
    unsafe { (base as *const XdShmHeader).read_unaligned() }
}

fn header_plausible(hdr: &XdShmHeader, map_len: usize) -> bool {
    if hdr.w == 0 || hdr.h == 0 || hdr.w > 16384 || hdr.h > 16384 {
        return false;
    }
    if hdr.w % 2 != 0 {
        return false;
    }
    let want = HEADER_SIZE + hdr.w as usize * hdr.h as usize * 4;
    // Require exact size: producer recreates the segment on dims change.
    want == map_len
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
        let hdr = XdShmHeader {
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
        unsafe {
            (ptr as *mut XdShmHeader).write_unaligned(hdr);
            let px = (ptr as *mut u8).add(HEADER_SIZE) as *mut f32;
            for i in 0..(w as usize * h as usize) {
                px.add(i).write((i as f32) / 1000.0);
            }
            // publish
            (ptr as *mut u32).add(2).write_volatile(2);
        }
        TestSeg {
            name: cname,
            ptr,
            len,
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
}
