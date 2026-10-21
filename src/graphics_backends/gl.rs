use super::{DepthFdRequest, DepthFdResult, GraphicsBackend};
use crate::depth_feed;
use derive_more::Deref;
use glutin_glx_sys::{
    Success,
    glx::{self, Glx},
};
use libc::{dlerror, dlopen, dlsym};
use log::{debug, info, warn};
use openvr as vr;
use openxr as xr;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, LazyLock, Once};
use std::time::{Duration, Instant};

static GLX: LazyLock<Library> = LazyLock::new(|| Library::new(c"libGLX.so.0"));

/// GL internal format for D32F imports.
const GL_DEPTH_COMPONENT32F: u32 = 0x8CAC;

pub struct GlData {
    session_data: Arc<SessionCreateInfo>,
    images: Vec<u32>,
    format: u32,
    read_fbo: u32,
    draw_fbo: u32,
    depth_images: Vec<u32>,
    depth_format: u32,
    /// GL_EXT_memory_object(_fd) + GL_EXT_semaphore(_fd) all present.
    fd_import_viable: bool,
    depth_fd: Option<DepthFdCache>,
    /// Last FD key whose import failed + earliest retry time. A failed
    /// import for an epoch never succeeds later (FDs are fixed per epoch),
    /// so retrying every submit just burns ~20 ms/frame on blocking GL
    /// errors. Skip until the key changes or the cooldown elapses.
    fd_failed_key: Option<(u32, u32, u32, i32)>,
    fd_next_retry: Option<Instant>,
}

/// Cooldown between import retries for the same failing key.
const FD_IMPORT_RETRY_COOLDOWN: Duration = Duration::from_secs(2);

#[derive(Copy, Clone, Default)]
struct DepthFdSlot {
    mem: u32,
    tex: u32,
    sem: u32,
}

struct DepthFdCache {
    epoch: u32,
    seg_len: usize,
    w: u32,
    h: u32,
    vk_format: i32,
    alloc_bytes: u32,
    slots: [DepthFdSlot; 3],
    /// (epoch, generation) of the last semaphore wait. Survives session restarts
    /// (same GL context) so a restart can never double-wait one signal.
    last_waited: Option<(u32, u64)>,
    /// Copies in flight: (fence address bits, generation). Integers keep the
    /// cache Send + Sync; cast back to GLsync at the use site. Acked on
    /// completion, never before — the producer may reuse a slot as soon as
    /// it sees the ack.
    pending: Vec<(u64, u64)>,
}

/// Required GL extensions for the FD import path.
const FD_IMPORT_EXTS: [&str; 4] = [
    "GL_EXT_memory_object",
    "GL_EXT_memory_object_fd",
    "GL_EXT_semaphore",
    "GL_EXT_semaphore_fd",
];

/// Opaque-fd handle type for memory/semaphore imports.
const HANDLE_TYPE_OPAQUE_FD: u32 = 0x9586;

type PfCreateMemoryObjects = unsafe extern "system" fn(gl::types::GLsizei, *mut gl::types::GLuint);
type PfDeleteMemoryObjects =
    unsafe extern "system" fn(gl::types::GLsizei, *const gl::types::GLuint);
type PfTextureStorageMem2D = unsafe extern "system" fn(
    gl::types::GLuint,
    gl::types::GLsizei,
    gl::types::GLenum,
    gl::types::GLsizei,
    gl::types::GLsizei,
    gl::types::GLuint,
    gl::types::GLuint64,
);
type PfImportMemoryFd = unsafe extern "system" fn(
    gl::types::GLuint,
    gl::types::GLuint64,
    gl::types::GLenum,
    gl::types::GLint,
);
type PfGenSemaphores = unsafe extern "system" fn(gl::types::GLsizei, *mut gl::types::GLuint);
type PfDeleteSemaphores = unsafe extern "system" fn(gl::types::GLsizei, *const gl::types::GLuint);
type PfImportSemaphoreFd =
    unsafe extern "system" fn(gl::types::GLuint, gl::types::GLenum, gl::types::GLint);
type PfWaitSemaphore = unsafe extern "system" fn(
    gl::types::GLuint,
    gl::types::GLuint,
    *const gl::types::GLuint,
    gl::types::GLuint,
    *const gl::types::GLuint,
    *const gl::types::GLenum,
);

/// EXT_memory_object(_fd) + EXT_semaphore(_fd) entry points. The `gl` crate
/// has no bindings for these, so they are loaded via `GetProcAddress` like
/// the core entry points. Every call site checks `Some`.
struct DepthExtFns {
    create_memory_objects: Option<PfCreateMemoryObjects>,
    delete_memory_objects: Option<PfDeleteMemoryObjects>,
    texture_storage_mem_2d: Option<PfTextureStorageMem2D>,
    import_memory_fd: Option<PfImportMemoryFd>,
    gen_semaphores: Option<PfGenSemaphores>,
    delete_semaphores: Option<PfDeleteSemaphores>,
    import_semaphore_fd: Option<PfImportSemaphoreFd>,
    wait_semaphore: Option<PfWaitSemaphore>,
}

impl DepthExtFns {
    fn ready(&self) -> bool {
        self.create_memory_objects.is_some()
            && self.delete_memory_objects.is_some()
            && self.texture_storage_mem_2d.is_some()
            && self.import_memory_fd.is_some()
            && self.gen_semaphores.is_some()
            && self.delete_semaphores.is_some()
            && self.import_semaphore_fd.is_some()
            && self.wait_semaphore.is_some()
    }
}

static DEPTH_EXT: LazyLock<DepthExtFns> = LazyLock::new(|| {
    let glx = Glx::load_with(|func| {
        let name = unsafe { CString::from_vec_unchecked(func.as_bytes().to_vec()) };
        GLX.get(&name)
    });
    unsafe fn proc<F>(glx: &Glx, name: &'static CStr) -> Option<F> {
        // SAFETY: GetProcAddress with a static name performs no GL work;
        // transmute matches the Khronos registry, call sites check Some.
        unsafe {
            let p = glx.GetProcAddress(name.as_ptr().cast::<u8>()) as *const c_void;
            if p.is_null() {
                None
            } else {
                Some(std::mem::transmute_copy::<*const c_void, F>(&p))
            }
        }
    }
    // SAFETY: as above; loading addresses performs no GL work.
    unsafe {
        DepthExtFns {
            create_memory_objects: proc(&glx, c"glCreateMemoryObjectsEXT"),
            delete_memory_objects: proc(&glx, c"glDeleteMemoryObjectsEXT"),
            texture_storage_mem_2d: proc(&glx, c"glTextureStorageMem2DEXT"),
            import_memory_fd: proc(&glx, c"glImportMemoryFdEXT"),
            gen_semaphores: proc(&glx, c"glGenSemaphoresEXT"),
            delete_semaphores: proc(&glx, c"glDeleteSemaphoresEXT"),
            import_semaphore_fd: proc(&glx, c"glImportSemaphoreFdEXT"),
            wait_semaphore: proc(&glx, c"glWaitSemaphoreEXT"),
        }
    }
});

/// Query the current context's extension set (GL 3.0+ enumeration with a
/// pre-3.0 fallback). Call only with a context current.
fn gl_extension_set() -> HashSet<String> {
    let mut set = HashSet::new();
    unsafe {
        if gl::GetStringi::is_loaded() {
            let mut n = 0;
            gl::GetIntegerv(gl::NUM_EXTENSIONS, &mut n);
            for i in 0..n {
                let s = gl::GetStringi(gl::EXTENSIONS, i as u32);
                if !s.is_null() {
                    set.insert(
                        CStr::from_ptr(s as *const c_char)
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        } else if gl::GetString::is_loaded() {
            let s = gl::GetString(gl::EXTENSIONS);
            if !s.is_null() {
                let all = CStr::from_ptr(s as *const c_char).to_string_lossy();
                set.extend(all.split_whitespace().map(str::to_owned));
            }
        }
    }
    set
}

#[derive(Deref)]
struct SessionCreateInfo(xr::opengl::SessionCreateInfo);
// SAFETY: SessionCreateInfo is only not Send + Sync because of the pointer next field.
// We don't even use this field so it's fine.
unsafe impl Send for SessionCreateInfo {}
unsafe impl Sync for SessionCreateInfo {}

unsafe fn get_fbconfig(
    glx: &Glx,
    display: *mut glx::types::Display,
    glx_context: glx::types::GLXContext,
) -> Option<glx::types::GLXFBConfig> {
    unsafe {
        let mut config_id = 0;
        let ret = glx.QueryContext(display, glx_context, glx::FBCONFIG_ID as _, &mut config_id);
        if ret != Success as i32 {
            warn!("Failed to get fbconfig id from context (error code {ret})");
            return None;
        }

        let mut screen = 0;
        let ret = glx.QueryContext(display, glx_context, glx::SCREEN as _, &mut screen);
        if ret != Success as i32 {
            warn!("Failed to get GLX screen for context (error code {ret})");
            return None;
        }

        let attrs = [glx::FBCONFIG_ID, config_id as _, glx::NONE];
        let mut items = 0;
        let cfgs = glx.ChooseFBConfig(display, screen, attrs.as_ptr() as _, &mut items);
        (!cfgs.is_null() && items >= 0).then(|| std::slice::from_raw_parts(cfgs, items as usize)[0])
    }
}

unsafe fn get_visualid(
    glx: &Glx,
    display: *mut glx::types::Display,
    cfg: Option<glx::types::GLXFBConfig>,
) -> Option<u32> {
    unsafe {
        let visual = glx.GetVisualFromFBConfig(display, cfg?);
        if visual.is_null() {
            warn!("No visual available from fbconfig.");
            None
        } else {
            Some((&raw const (*visual).visualid).read() as u32)
        }
    }
}

impl GlData {
    // Returns None if we couldn't get the display.
    pub(crate) fn new() -> Option<Self> {
        let glx = Glx::load_with(|func| {
            let func = unsafe { CString::from_vec_unchecked(func.as_bytes().to_vec()) };
            GLX.get(&func)
        });

        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            gl::load_with(|f| {
                let f = unsafe { CString::from_vec_unchecked(f.as_bytes().to_vec()) };
                unsafe { glx.GetProcAddress(f.as_ptr().cast()) }.cast()
            });

            if log::log_enabled!(log::Level::Debug) {
                unsafe {
                    gl::DebugMessageCallback(Some(debug_message), std::ptr::null());
                    gl::Enable(gl::DEBUG_OUTPUT);
                }
            }
        });

        // Grab the session info on creation - this makes us resilient against session restarts,
        // which could result in us trying to grab the context from a different thread
        let session_info = unsafe {
            let x_display = glx.GetCurrentDisplay();
            if x_display.is_null() {
                warn!("X display is null!");
                return None;
            }
            let glx_context = glx.GetCurrentContext();
            if glx_context.is_null() {
                warn!("GLX context was null!");
                return None;
            }

            let glx_drawable = glx.GetCurrentDrawable();
            let fbconfig = get_fbconfig(&glx, x_display, glx_context);
            let visualid = get_visualid(&glx, x_display, fbconfig);

            xr::opengl::SessionCreateInfo::Xlib {
                x_display: x_display.cast(),
                glx_fb_config: fbconfig.map(|p| p.cast_mut()).unwrap_or_else(|| {
                    warn!("No fbconfig found.");
                    1 as _
                }),
                visualid: visualid.unwrap_or(1),
                glx_drawable,
                glx_context: glx_context.cast_mut(),
            }
        };

        let mut fbos = [0; 2];
        unsafe {
            gl::GenFramebuffers(fbos.len() as i32, fbos.as_mut_ptr());
        }

        // FD import viability probe: extension strings + entry points both
        // required. Logged once per process; decides whether FD depth runs.
        let exts = gl_extension_set();
        let mut fd_viable = true;
        for e in FD_IMPORT_EXTS {
            let present = exts.contains(e);
            info!("depth FD import ext {e}: {present}");
            fd_viable &= present;
        }
        info!("depth FD import viable: {fd_viable}");

        Some(GlData {
            session_data: Arc::new(SessionCreateInfo(session_info)),
            images: Default::default(),
            format: 0,
            read_fbo: fbos[0],
            draw_fbo: fbos[1],
            depth_images: Default::default(),
            depth_format: 0,
            fd_import_viable: fd_viable,
            depth_fd: None,
            fd_failed_key: None,
            fd_next_retry: None,
        })
    }
}

impl GraphicsBackend for GlData {
    type Api = xr::OpenGL;
    type OpenVrTexture = gl::types::GLuint;
    type NiceFormat = u32;

    fn to_nice_format(format: u32) -> Self::NiceFormat {
        format
    }

    fn session_create_info(&self) -> <Self::Api as openxr::Graphics>::SessionCreateInfo {
        // SAFETY: SessionCreateInfo should be Copy anyway but doesn't work right
        // https://github.com/Ralith/openxrs/issues/183
        unsafe { std::ptr::read(&**self.session_data) }
    }

    #[inline]
    fn get_texture(texture: &vr::Texture_t) -> Option<Self::OpenVrTexture> {
        Some(texture.handle as _)
    }

    #[inline]
    fn store_swapchain_images(
        &mut self,
        images: Vec<<Self::Api as xr::Graphics>::SwapchainImage>,
        format: u32,
    ) {
        self.images = images;
        self.format = format;
    }

    #[inline]
    fn swapchain_info_for_texture(
        &self,
        texture: Self::OpenVrTexture,
        bounds: vr::VRTextureBounds_t,
        color_space: vr::EColorSpace,
    ) -> xr::SwapchainCreateInfo<Self::Api> {
        let mut fmt = 0;
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_INTERNAL_FORMAT, &mut fmt);
        }
        let xr::Rect2Di { extent, .. } = texture_rect_from_bounds(texture, bounds);

        let fmt = match color_space {
            vr::EColorSpace::Linear => fmt,
            vr::EColorSpace::Gamma | vr::EColorSpace::Auto => match fmt as gl::types::GLenum {
                gl::RGBA8 => gl::SRGB8_ALPHA8 as gl::types::GLint,
                gl::RGB8 => gl::SRGB8 as gl::types::GLint,
                _ => fmt,
            },
        };

        xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::TRANSFER_DST,
            format: fmt as u32,
            sample_count: 1,
            width: extent.width as u32,
            height: extent.height as u32,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        }
    }

    fn copy_texture_to_swapchain(
        &self,
        eye: vr::EVREye,
        texture: Self::OpenVrTexture,
        _color_space: vr::EColorSpace,
        bounds: vr::VRTextureBounds_t,
        image_index: usize,
        _submit_flags: vr::EVRSubmitFlags,
    ) -> xr::Extent2Di {
        let swapchain_texture = self.images[image_index];

        let xr::Rect2Di { extent, offset } = texture_rect_from_bounds(texture, bounds);

        let mut fmt = 0;
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_INTERNAL_FORMAT, &mut fmt);
            gl::BindTexture(gl::TEXTURE_2D, 0);
        }

        if self.format == fmt as u32 {
            unsafe {
                gl::CopyImageSubData(
                    texture,
                    gl::TEXTURE_2D,
                    0, // level
                    offset.x,
                    offset.y,
                    0, // z
                    swapchain_texture,
                    gl::TEXTURE_2D_ARRAY,
                    0, // x
                    0, // y
                    0, // z
                    eye as i32,
                    extent.width,
                    extent.height,
                    1,
                );
            }
        } else {
            unsafe {
                gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
                gl::FramebufferTexture2D(
                    gl::READ_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    texture,
                    0,
                );
                gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_fbo);
                gl::FramebufferTextureLayer(
                    gl::DRAW_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    swapchain_texture,
                    0,
                    eye as i32,
                );

                gl::BlitFramebuffer(
                    offset.x,
                    offset.y,
                    offset.x + extent.width,
                    offset.y + extent.height,
                    0,
                    0,
                    extent.width,
                    extent.height,
                    gl::COLOR_BUFFER_BIT,
                    gl::NEAREST,
                );

                gl::BindFramebuffer(gl::READ_FRAMEBUFFER, 0);
                gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
            }
        }

        extent
    }

    fn copy_overlay_to_swapchain(
        &mut self,
        texture: Self::OpenVrTexture,
        bounds: openvr::VRTextureBounds_t,
        image_index: usize,
    ) -> openxr::Extent2Di {
        self.copy_texture_to_swapchain(
            vr::EVREye::Left,
            texture,
            vr::EColorSpace::Auto,
            bounds,
            image_index,
            vr::EVRSubmitFlags::Default,
        )
    }

    fn supports_depth() -> bool {
        true
    }

    fn depth_format_for_header(vk_format: i32) -> Option<<Self::Api as xr::Graphics>::Format> {
        depth_gl_format(vk_format)
    }

    fn depth_fd_import_viable(&self) -> bool {
        // Extension string probe (logged at session init) plus the loaded
        // entry points themselves; both must hold to attempt imports.
        self.fd_import_viable && DEPTH_EXT.ready()
    }

    fn import_and_copy_depth_fd(&mut self, req: &DepthFdRequest) -> DepthFdResult {
        let inert = || DepthFdResult {
            copied: false,
            ack_gen: None,
        };
        if !self.fd_import_viable {
            return inert();
        }
        // D32F only in prod: 4 bytes/texel.
        let Some(bpp) = depth_texel_bytes(req.vk_format) else {
            return inert();
        };
        let Some(internal) = depth_gl_format(req.vk_format) else {
            return inert();
        };
        if req.generation == 0 || req.eye_w == 0 || req.eye_h == 0 {
            return inert();
        }
        // Stdin/stdout/stderr are never valid depth FDs; the shim's
        // pre-publish sentinel is -1.
        if req.mem_fds.iter().any(|&f| f <= 2) || req.sem_fds.iter().any(|&f| f <= 2) {
            return inert();
        }
        if self.depth_images.is_empty() || req.depth_image_index >= self.depth_images.len() {
            return inert();
        }
        // alloc_bytes bound: the driver allocation can pad past w*h*bpp,
        // so imports cover the full allocation — but a requirement larger
        // than a known allocation means writer/reader drift. Reject before
        // any GL call (0 = unknown, tolerate).
        let want = req.w as u64 * req.h as u64 * bpp;
        if req.alloc_bytes != 0 && want > req.alloc_bytes as u64 {
            warn!(
                "depth FD setup rejected: want {want} > alloc {} (epoch {}, {}x{} fmt {})",
                req.alloc_bytes, req.setup_epoch, req.w, req.h, req.vk_format,
            );
            return inert();
        }
        let dst = self.depth_images[req.depth_image_index];
        // Reimport on epoch/dims/format change (old GL objects first).
        // Segment size is deliberately NOT part of the identity.
        let key = (req.setup_epoch, req.w, req.h, req.vk_format);
        let cached_key = self
            .depth_fd
            .as_ref()
            .map(|c| (c.epoch, c.w, c.h, c.vk_format));
        if cached_key != Some(key) {
            // Failed imports never heal for this key (FDs are fixed per
            // epoch): skip the ~20 ms of blocking GL errors until the key
            // changes or the cooldown elapses. Without this a broken FD
            // path halves the framerate retrying every submit.
            if self.fd_failed_key == Some(key) {
                let retry_ok = self.fd_next_retry.is_none_or(|t| Instant::now() >= t);
                if !retry_ok {
                    return inert();
                }
            }
            match self.depth_fd.as_ref() {
                Some(c) => debug!(
                    "depth FD reimport: epoch {}→{}, dims {}x{}→{}x{}, format {}→{}, seg {}→{}, alloc {}→{}",
                    c.epoch,
                    req.setup_epoch,
                    c.w,
                    c.h,
                    req.w,
                    req.h,
                    c.vk_format,
                    req.vk_format,
                    c.seg_len,
                    req.seg_len,
                    c.alloc_bytes,
                    req.alloc_bytes,
                ),
                None => debug!(
                    "depth FD import: epoch {}, dims {}x{}, format {}, seg {}, alloc {}",
                    req.setup_epoch, req.w, req.h, req.vk_format, req.seg_len, req.alloc_bytes,
                ),
            }
            self.drop_depth_imports();
            match Self::import_fd_slots(req, bpp, internal) {
                Some(slots) => {
                    self.depth_fd = Some(DepthFdCache {
                        epoch: req.setup_epoch,
                        seg_len: req.seg_len,
                        w: req.w,
                        h: req.h,
                        vk_format: req.vk_format,
                        alloc_bytes: req.alloc_bytes,
                        slots,
                        last_waited: None,
                        pending: Vec::new(),
                    });
                    // Healed (or first success): clear any failure record.
                    self.fd_failed_key = None;
                    self.fd_next_retry = None;
                }
                None => {
                    // Remember the failure so subsequent submits for this key
                    // skip the import until the cooldown elapses. The warn
                    // inside import_fd_slots already named the failing step.
                    self.fd_failed_key = Some(key);
                    self.fd_next_retry = Some(Instant::now() + FD_IMPORT_RETRY_COOLDOWN);
                    return inert();
                }
            }
        }
        // Drain completed copies; ack only fully-copied generations so the
        // producer never reuses a slot whose copy is still in flight.
        let mut ack = None;
        if let Some(cache) = self.depth_fd.as_mut() {
            let mut i = 0;
            while i < cache.pending.len() {
                let (bits, g) = cache.pending[i];
                let sync = bits as gl::types::GLsync;
                let status = unsafe { gl::ClientWaitSync(sync, 0, 0) };
                if status == gl::ALREADY_SIGNALED || status == gl::CONDITION_SATISFIED {
                    unsafe { gl::DeleteSync(sync) };
                    cache.pending.swap_remove(i);
                    ack = Some(ack.map_or(g, |a: u64| a.max(g)));
                } else if status == gl::WAIT_FAILED {
                    unsafe { gl::DeleteSync(sync) };
                    cache.pending.swap_remove(i);
                } else {
                    i += 1;
                }
            }
            // Bound the ack trail: the just-submitted copy is almost always
            // done long before the next submit, so a short bounded wait lets
            // the ack track the newest generation instead of trailing a full
            // frame. Long waits only happen under real GPU pressure, where
            // slowing submissions is the correct backpressure anyway.
            if let Some(&(bits, g)) = cache.pending.last() {
                let sync = bits as gl::types::GLsync;
                let status =
                    unsafe { gl::ClientWaitSync(sync, gl::SYNC_FLUSH_COMMANDS_BIT, 5_000_000) };
                if status == gl::ALREADY_SIGNALED || status == gl::CONDITION_SATISFIED {
                    unsafe { gl::DeleteSync(sync) };
                    cache.pending.pop();
                    ack = Some(ack.map_or(g, |a: u64| a.max(g)));
                } else if status == gl::WAIT_FAILED {
                    unsafe { gl::DeleteSync(sync) };
                    cache.pending.pop();
                }
            }
        }
        let Some(cache) = self.depth_fd.as_mut() else {
            return DepthFdResult {
                copied: false,
                ack_gen: ack,
            };
        };
        // Wait once per generation (binary semaphores; skip on re-copy).
        let slot = (req.generation % 3) as usize;
        if cache.last_waited != Some((req.setup_epoch, req.generation)) {
            let Some(wait) = DEPTH_EXT.wait_semaphore else {
                return DepthFdResult {
                    copied: false,
                    ack_gen: ack,
                };
            };
            unsafe {
                wait(
                    cache.slots[slot].sem,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
            cache.last_waited = Some((req.setup_epoch, req.generation));
        }
        let ok = unsafe {
            for eye in 0..2u32 {
                gl::CopyImageSubData(
                    cache.slots[slot].tex,
                    gl::TEXTURE_2D,
                    0,
                    (eye * req.eye_w) as i32,
                    0,
                    0,
                    dst,
                    gl::TEXTURE_2D_ARRAY,
                    0,
                    0,
                    0,
                    eye as i32,
                    req.eye_w as i32,
                    req.eye_h as i32,
                    1,
                );
            }
            gl::GetError() == gl::NO_ERROR
        };
        if !ok {
            return DepthFdResult {
                copied: false,
                ack_gen: ack,
            };
        }
        let sync = unsafe { gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0) };
        if !sync.is_null() {
            cache.pending.push((sync as u64, req.generation));
            unsafe { gl::Flush() };
        }
        DepthFdResult {
            copied: true,
            ack_gen: ack,
        }
    }

    fn drop_depth_imports(&mut self) {
        if let Some(cache) = self.depth_fd.take() {
            unsafe {
                for (bits, _) in cache.pending {
                    gl::DeleteSync(bits as gl::types::GLsync);
                }
                for s in &cache.slots {
                    if let Some(f) = DEPTH_EXT.delete_semaphores {
                        f(1, &s.sem);
                    }
                    gl::DeleteTextures(1, &s.tex);
                    if let Some(f) = DEPTH_EXT.delete_memory_objects {
                        f(1, &s.mem);
                    }
                }
            }
        }
    }

    fn depth_swapchain_create_info(
        &self,
        color_info: &xr::SwapchainCreateInfo<Self::Api>,
        depth_format: <Self::Api as xr::Graphics>::Format,
    ) -> Option<xr::SwapchainCreateInfo<Self::Api>> {
        if color_info.width == 0 || color_info.height == 0 {
            return None;
        }
        Some(xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | xr::SwapchainUsageFlags::TRANSFER_DST,
            format: depth_format,
            sample_count: 1,
            width: color_info.width,
            height: color_info.height,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })
    }

    fn store_depth_swapchain_images(
        &mut self,
        images: Vec<<Self::Api as xr::Graphics>::SwapchainImage>,
        format: u32,
    ) {
        self.depth_images = images;
        self.depth_format = format;
    }
}

/// Texel footprint for the SHM Vulkan depth format: D32F-only in prod.
fn depth_texel_bytes(vk_format: i32) -> Option<u64> {
    match vk_format {
        depth_feed::FORMAT_D32F => Some(4),
        _ => None,
    }
}

/// GL internal format for the SHM Vulkan depth format: D32F-only.
fn depth_gl_format(vk_format: i32) -> Option<u32> {
    match vk_format {
        depth_feed::FORMAT_D32F => Some(GL_DEPTH_COMPONENT32F),
        _ => None,
    }
}

impl GlData {
    /// Check + drain the GL error flag. Returns true when clean.
    /// Call sites check Some, so a logged string here is the exact
    /// attribution for async debug-callback storms.
    unsafe fn gl_ok() -> bool {
        unsafe { gl::GetError() == gl::NO_ERROR }
    }

    /// Import all 3 ring slots (memory object + texture storage + binary
    /// semaphore each). The FDs are borrowed, never closed or duplicated:
    /// same process, producer-held, valid for the segment lifetime.
    /// Never declares DEDICATED (drivers reject it on some paths).
    fn import_fd_slots(req: &DepthFdRequest, bpp: u64, internal: u32) -> Option<[DepthFdSlot; 3]> {
        // Driver allocation can pad past w*h*bpp (alignment). Import the
        // full allocation so glImportMemoryFdEXT never sees a truncated
        // object. The want > alloc rejection already ran in the caller;
        // alloc==0 means the producer predates the header field.
        let want = req.w as u64 * req.h as u64 * bpp;
        let size = want.max(req.alloc_bytes as u64);
        if size == 0 {
            return None;
        }
        let create_mem = DEPTH_EXT.create_memory_objects?;
        let import_mem = DEPTH_EXT.import_memory_fd?;
        let tex_storage = DEPTH_EXT.texture_storage_mem_2d?;
        let gen_sem = DEPTH_EXT.gen_semaphores?;
        let import_sem = DEPTH_EXT.import_semaphore_fd?;
        // Drain pre-existing errors so the first failure below attributes
        // exactly (debug callbacks are async; without this the blame lands
        // on whatever call happens to poll GetError next).
        unsafe { while gl::GetError() != gl::NO_ERROR {} }
        let mut slots = [DepthFdSlot::default(); 3];
        for i in 0..3 {
            let (mut mem, mut tex, mut sem) = (0, 0, 0);
            let mut failed_at: Option<&'static str> = None;
            unsafe {
                create_mem(1, &mut mem);
                if !Self::gl_ok() || mem == 0 {
                    failed_at = Some("glCreateMemoryObjectsEXT");
                } else {
                    import_mem(mem, size, HANDLE_TYPE_OPAQUE_FD, req.mem_fds[i]);
                    if !Self::gl_ok() {
                        failed_at = Some("glImportMemoryFdEXT");
                    } else {
                        gl::GenTextures(1, &mut tex);
                        gl::BindTexture(gl::TEXTURE_2D, tex);
                        tex_storage(tex, 1, internal, req.w as i32, req.h as i32, mem, 0);
                        if !Self::gl_ok() || tex == 0 {
                            failed_at = Some("glTextureStorageMem2DEXT");
                        } else {
                            gl::BindTexture(gl::TEXTURE_2D, 0);
                            gen_sem(1, &mut sem);
                            if !Self::gl_ok() || sem == 0 {
                                failed_at = Some("glGenSemaphoresEXT");
                            } else {
                                import_sem(sem, HANDLE_TYPE_OPAQUE_FD, req.sem_fds[i]);
                                if !Self::gl_ok() {
                                    failed_at = Some("glImportSemaphoreFdEXT");
                                }
                            }
                        }
                    }
                }
                // Final sweep: catches errors the per-call polls missed
                // (async delivery) while attribution above names the step.
                if failed_at.is_none() && !Self::gl_ok() {
                    failed_at = Some("trailing (async)");
                }
            }
            if let Some(step) = failed_at {
                warn!(
                    "depth FD import failed at {step} (slot {i}/3, epoch {}, {}x{} fmt {} memfd {} semfd {} size {})",
                    req.setup_epoch,
                    req.w,
                    req.h,
                    req.vk_format,
                    req.mem_fds[i],
                    req.sem_fds[i],
                    size,
                );
                unsafe {
                    Self::delete_fd_slot(&DepthFdSlot { mem, tex, sem });
                    for s in slots.iter().take(i) {
                        Self::delete_fd_slot(s);
                    }
                }
                return None;
            }
            slots[i] = DepthFdSlot { mem, tex, sem };
        }
        Some(slots)
    }

    unsafe fn delete_fd_slot(s: &DepthFdSlot) {
        unsafe {
            if s.sem != 0
                && let Some(f) = DEPTH_EXT.delete_semaphores
            {
                f(1, &s.sem);
            }
            if s.tex != 0 {
                gl::DeleteTextures(1, &s.tex);
            }
            if s.mem != 0
                && let Some(f) = DEPTH_EXT.delete_memory_objects
            {
                f(1, &s.mem);
            }
        }
    }
}

fn texture_rect_from_bounds(
    texture: glx::types::GLuint,
    bounds: vr::VRTextureBounds_t,
) -> xr::Rect2Di {
    let [mut height, mut width] = Default::default();
    unsafe {
        gl::BindTexture(gl::TEXTURE_2D, texture);
        gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_WIDTH, &mut width);
        gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_HEIGHT, &mut height);
        gl::BindTexture(gl::TEXTURE_2D, 0);
    }
    let width_min = bounds.uMin * width as f32;
    let width_max = bounds.uMax * width as f32;
    let height_min = bounds.vMin * height as f32;
    let height_max = bounds.vMax * height as f32;

    xr::Rect2Di {
        extent: xr::Extent2Di {
            width: (width_max - width_min).abs() as i32,
            height: (height_max - height_min).abs() as i32,
        },
        offset: xr::Offset2Di {
            x: width_min.min(width_max) as i32,
            y: height_min.min(height_max) as i32,
        },
    }
}

extern "system" fn debug_message(
    source: gl::types::GLenum,
    ty: gl::types::GLenum,
    id: gl::types::GLuint,
    severity: gl::types::GLenum,
    _: gl::types::GLsizei,
    message: *const c_char,
    _: *mut c_void,
) {
    let source = match source {
        gl::DEBUG_SOURCE_API => "OpenGL Api",
        gl::DEBUG_SOURCE_OTHER => "Other",
        _ => "<unknown>",
    };

    let ty = match ty {
        gl::DEBUG_TYPE_ERROR => "Error",
        gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated Behavior",
        gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
        gl::DEBUG_TYPE_PORTABILITY => "Portability Issue",
        gl::DEBUG_TYPE_PERFORMANCE => "Performance Issue",
        gl::DEBUG_TYPE_OTHER => "Other",
        _ => "<unknown>",
    };

    let severity = match severity {
        gl::DEBUG_SEVERITY_HIGH => "High",
        gl::DEBUG_SEVERITY_MEDIUM => "Medium",
        gl::DEBUG_SEVERITY_LOW => "Low",
        gl::DEBUG_SEVERITY_NOTIFICATION => "Notification",
        _ => "<unknown>",
    };
    let message = unsafe { CStr::from_ptr(message) };
    log::debug!("(severity: {severity}, id: {id}) {ty} message from {source}: {message:?}");
}

struct Library(*mut c_void);
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Library {
    fn new(name: &CStr) -> Self {
        let handle = unsafe { dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if handle.is_null() {
            let err = unsafe { CStr::from_ptr(dlerror()) };
            panic!("Failed to load {name:?}: {err:?}");
        }

        Self(handle)
    }

    fn get(&self, function: &CStr) -> *const c_void {
        // clear old error
        unsafe {
            dlerror();
        }

        let symbol = unsafe { dlsym(self.0, function.as_ptr()) };
        if symbol.is_null() {
            let err = unsafe { dlerror() };
            if !err.is_null() {
                panic!("Failed to get symbol {function:?}: {:?}", unsafe {
                    CStr::from_ptr(err)
                });
            }
        }
        symbol
    }
}

#[cfg(test)]
mod tests {
    use super::{GL_DEPTH_COMPONENT32F, depth_gl_format, depth_texel_bytes};
    use crate::depth_feed::FORMAT_D32F;

    #[test]
    fn depth_format_maps_d32f_only() {
        assert_eq!(depth_texel_bytes(FORMAT_D32F), Some(4));
        assert_eq!(depth_texel_bytes(96), None);
        assert_eq!(depth_texel_bytes(130), None);
        assert_eq!(depth_gl_format(FORMAT_D32F), Some(GL_DEPTH_COMPONENT32F));
        assert_eq!(depth_gl_format(96), None);
        assert_eq!(depth_gl_format(130), None);
    }
}
