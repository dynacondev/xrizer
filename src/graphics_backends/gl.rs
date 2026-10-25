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
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, LazyLock, Once};
use std::time::{Duration, Instant};

static GLX: LazyLock<Library> = LazyLock::new(|| Library::new(c"libGLX.so.0"));

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
    /// so back off instead of paying blocking GL errors every frame.
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
    w: u32,
    h: u32,
    vk_format: i32,
    alloc_bytes: u32,
    slots: [DepthFdSlot; 2],
    /// (epoch, generation) of the last semaphore wait. Waits are issued
    /// once per generation (binary-semaphore semantics); a re-copy of an
    /// already-waited generation skips straight to the copy.
    last_waited: Option<(u32, u64)>,
    /// In-flight copy completion fence: (fence bits, generation). At most
    /// one — the caller holds at most one SHM pin, so at most one copy is
    /// ever outstanding. Polled (zero timeout) before the next pin.
    pending: Option<(u64, u64)>,
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

type PfCreateMemoryObjects =
    unsafe extern "system" fn(n: gl::types::GLsizei, memories: *mut gl::types::GLuint);
type PfDeleteMemoryObjects =
    unsafe extern "system" fn(n: gl::types::GLsizei, memories: *const gl::types::GLuint);
type PfImportMemoryFd = unsafe extern "system" fn(
    memory: gl::types::GLuint,
    size: u64,
    handle_type: gl::types::GLenum,
    fd: libc::c_int,
);
type PfTextureStorageMem2D = unsafe extern "system" fn(
    texture: gl::types::GLuint,
    levels: gl::types::GLsizei,
    internal_format: gl::types::GLenum,
    width: gl::types::GLsizei,
    height: gl::types::GLsizei,
    memory: gl::types::GLuint,
    offset: u64,
);
type PfGenSemaphores =
    unsafe extern "system" fn(n: gl::types::GLsizei, semaphores: *mut gl::types::GLuint);
type PfDeleteSemaphores =
    unsafe extern "system" fn(n: gl::types::GLsizei, semaphores: *const gl::types::GLuint);
type PfImportSemaphoreFd = unsafe extern "system" fn(
    semaphore: gl::types::GLuint,
    handle_type: gl::types::GLenum,
    fd: libc::c_int,
);
type PfWaitSemaphore = unsafe extern "system" fn(
    semaphore: gl::types::GLuint,
    num_buffer_barriers: gl::types::GLuint,
    buffers: *const gl::types::GLuint,
    num_texture_barriers: gl::types::GLuint,
    textures: *const gl::types::GLuint,
    src_layouts: *const gl::types::GLenum,
);

/// EXT_memory_object(_fd) + EXT_semaphore(_fd) entry points. Loaded once;
/// any missing entry disables the FD path (never a partial import).
struct DepthExtFns {
    create_memory: Option<PfCreateMemoryObjects>,
    delete_memory: Option<PfDeleteMemoryObjects>,
    import_memory_fd: Option<PfImportMemoryFd>,
    texture_storage_mem_2d: Option<PfTextureStorageMem2D>,
    gen_semaphores: Option<PfGenSemaphores>,
    delete_semaphores: Option<PfDeleteSemaphores>,
    import_semaphore_fd: Option<PfImportSemaphoreFd>,
    wait_semaphore: Option<PfWaitSemaphore>,
}

impl DepthExtFns {
    fn ready(&self) -> bool {
        self.create_memory.is_some()
            && self.delete_memory.is_some()
            && self.import_memory_fd.is_some()
            && self.texture_storage_mem_2d.is_some()
            && self.gen_semaphores.is_some()
            && self.delete_semaphores.is_some()
            && self.import_semaphore_fd.is_some()
            && self.wait_semaphore.is_some()
    }
}

fn load_ext_fn<T>(glx: &Glx, name: &CStr) -> Option<T> {
    let ptr = unsafe { glx.GetProcAddress(name.as_ptr().cast()) };
    if ptr.is_null() {
        return None;
    }
    // Bit-copy the word into the fn pointer: both are single words, and a
    // null check above excludes the only invalid bit pattern.
    Some(unsafe { (std::ptr::from_ref(&ptr) as *const T).read() })
}

static DEPTH_EXT: LazyLock<DepthExtFns> = LazyLock::new(|| {
    let glx = Glx::load_with(|func| {
        let func = unsafe { CString::from_vec_unchecked(func.as_bytes().to_vec()) };
        GLX.get(&func)
    });
    DepthExtFns {
        create_memory: load_ext_fn(&glx, c"glCreateMemoryObjectsEXT"),
        delete_memory: load_ext_fn(&glx, c"glDeleteMemoryObjectsEXT"),
        import_memory_fd: load_ext_fn(&glx, c"glImportMemoryFdEXT"),
        texture_storage_mem_2d: load_ext_fn(&glx, c"glTextureStorageMem2DEXT"),
        gen_semaphores: load_ext_fn(&glx, c"glGenSemaphoresEXT"),
        delete_semaphores: load_ext_fn(&glx, c"glDeleteSemaphoresEXT"),
        import_semaphore_fd: load_ext_fn(&glx, c"glImportSemaphoreFdEXT"),
        wait_semaphore: load_ext_fn(&glx, c"glWaitSemaphoreEXT"),
    }
});

/// True when the named extension is advertised on the current context.
fn gl_extension_present(name: &str) -> bool {
    unsafe {
        let mut count = 0;
        gl::GetIntegerv(gl::NUM_EXTENSIONS, &mut count);
        for i in 0..count {
            let s = gl::GetStringi(gl::EXTENSIONS, i as u32);
            if s.is_null() {
                continue;
            }
            if CStr::from_ptr(s.cast()).to_bytes() == name.as_bytes() {
                return true;
            }
        }
    }
    false
}

/// True when a GLX context is current on this thread. Free function (no
/// context needed to ask): resolves `glXGetCurrentContext` once via the
/// already-loaded libGLX and calls it directly.
fn glx_current_context() -> bool {
    type PfGetCurrentContext = unsafe extern "system" fn() -> glx::types::GLXContext;
    static GET_CURRENT: LazyLock<Option<PfGetCurrentContext>> = LazyLock::new(|| {
        let ptr = GLX.get(c"glXGetCurrentContext");
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { (std::ptr::from_ref(&ptr) as *const PfGetCurrentContext).read() })
    });
    match *GET_CURRENT {
        Some(f) => unsafe { !f().is_null() },
        // libGLX without GetCurrentContext is not a GLX we can work with;
        // fail closed (depth off) rather than issuing blind GL calls.
        None => false,
    }
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
        // Runs here (not lazily) because it needs a current context, which
        // creation guarantees but later calls do not.
        let mut fd_viable = true;
        for e in FD_IMPORT_EXTS {
            let present = gl_extension_present(e);
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

    fn supports_depth() -> bool {
        true
    }

    fn depth_format_for_header(vk_format: i32) -> Option<<Self::Api as xr::Graphics>::Format> {
        depth_gl_format(vk_format)
    }

    fn depth_fd_import_viable(&self) -> bool {
        // Extension strings were probed at creation (needs a context); entry
        // points resolve without one. Both must hold to attempt imports.
        self.fd_import_viable && DEPTH_EXT.ready()
    }

    fn depth_context_current(&self) -> bool {
        glx_current_context()
    }

    fn import_and_copy_depth_fd(&mut self, req: &DepthFdRequest) -> DepthFdResult {
        let inert = || DepthFdResult { copied: false };
        if !self.depth_fd_import_viable() {
            return inert();
        }
        let Some(bpp) = depth_texel_bytes(req.vk_format) else {
            return inert();
        };
        let Some(internal) = depth_gl_format(req.vk_format) else {
            return inert();
        };
        if req.generation == 0 || req.eye_w == 0 || req.eye_h == 0 {
            return inert();
        }
        // Stdin/stdout/stderr are never valid depth FDs; the producer's
        // pre-publish sentinel is -1.
        if req.mem_fds.iter().any(|&f| f <= 2) || req.sem_fds.iter().any(|&f| f <= 2) {
            return inert();
        }
        if self.depth_images.is_empty() || req.depth_image_index >= self.depth_images.len() {
            return inert();
        }
        let dst = self.depth_images[req.depth_image_index];
        // Reimport on epoch/dims/format change (old GL objects first).
        // Segment size is deliberately NOT part of the identity.
        let key = (req.epoch, req.w, req.h, req.vk_format);
        let cached_key = self
            .depth_fd
            .as_ref()
            .map(|c| (c.epoch, c.w, c.h, c.vk_format));
        if cached_key != Some(key) {
            // Failed imports never heal for this key (FDs are fixed per
            // epoch): skip the blocking GL errors until the key changes or
            // the cooldown elapses.
            if self.fd_failed_key == Some(key) {
                let retry_ok = self.fd_next_retry.is_none_or(|t| Instant::now() >= t);
                if !retry_ok {
                    return inert();
                }
            }
            // alloc_bytes bound: the driver allocation can pad past w*h*bpp,
            // so imports cover the full allocation — but a requirement
            // larger than a known allocation means writer/reader drift.
            // 0 = unknown, tolerate.
            let want = req.w as u64 * req.h as u64 * bpp;
            if req.alloc_bytes != 0 && want > req.alloc_bytes as u64 {
                warn!(
                    "depth FD setup rejected: want {want} > alloc {} (epoch {}, {}x{} fmt {})",
                    req.alloc_bytes, req.epoch, req.w, req.h, req.vk_format,
                );
                self.fd_failed_key = Some(key);
                self.fd_next_retry = Some(Instant::now() + FD_IMPORT_RETRY_COOLDOWN);
                return inert();
            }
            match self.depth_fd.as_ref() {
                Some(c) => debug!(
                    "depth FD reimport: epoch {}->{}, dims {}x{}->{}x{}, format {}->{}, alloc {}->{}",
                    c.epoch,
                    req.epoch,
                    c.w,
                    c.h,
                    req.w,
                    req.h,
                    c.vk_format,
                    req.vk_format,
                    c.alloc_bytes,
                    req.alloc_bytes,
                ),
                None => debug!(
                    "depth FD import: epoch {}, dims {}x{}, format {}, alloc {}",
                    req.epoch, req.w, req.h, req.vk_format, req.alloc_bytes,
                ),
            }
            self.drop_depth_imports();
            match Self::import_fd_slots(req, bpp, internal) {
                Some(slots) => {
                    self.depth_fd = Some(DepthFdCache {
                        epoch: req.epoch,
                        w: req.w,
                        h: req.h,
                        vk_format: req.vk_format,
                        alloc_bytes: req.alloc_bytes,
                        slots,
                        last_waited: None,
                        pending: None,
                    });
                    self.fd_failed_key = None;
                    self.fd_next_retry = None;
                }
                None => {
                    self.fd_failed_key = Some(key);
                    self.fd_next_retry = Some(Instant::now() + FD_IMPORT_RETRY_COOLDOWN);
                    return inert();
                }
            }
        }
        let Some(cache) = self.depth_fd.as_mut() else {
            return inert();
        };
        // Wait once per generation (binary-semaphore semantics; skip on
        // re-copy). Recorded only on a clean wait: a failed wait must not
        // mark the generation consumed, or the next copy would skip its
        // wait and risk tearing. The caller holds the slot's SHM pin across
        // this call, so the slot cannot be recycled under the wait.
        let slot = (req.generation % 2) as usize;
        if cache.last_waited != Some((req.epoch, req.generation)) {
            let Some(wait) = DEPTH_EXT.wait_semaphore else {
                return inert();
            };
            let waited_ok = unsafe {
                wait(
                    cache.slots[slot].sem,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                );
                gl::GetError() == gl::NO_ERROR
            };
            if !waited_ok {
                // Unsynchronized copy would risk tearing: skip the copy.
                // Next frame re-waits properly.
                return inert();
            }
            cache.last_waited = Some((req.epoch, req.generation));
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
            return inert();
        }
        let sync = unsafe { gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0) };
        if sync.is_null() {
            // Fence failed: the copy is queued but untracked. Report
            // success (pixels will land) while leaving completion polling
            // idle — the caller unpins on the next successful copy's fence.
            // Rare enough to warn.
            warn!(
                "depth FD fence creation failed (generation {})",
                req.generation
            );
            return DepthFdResult { copied: true };
        }
        if let Some((old_bits, old_gen)) = cache.pending.replace((sync as u64, req.generation)) {
            // Previous copy still untracked (caller only pins one slot, so
            // this means the prior fence was never polled): drop it rather
            // than leak. Deleting a sync never cancels queued work.
            unsafe { gl::DeleteSync(old_bits as gl::types::GLsync) };
            debug!("depth FD fence superseded (was gen {old_gen})");
        }
        unsafe { gl::Flush() };
        DepthFdResult { copied: true }
    }

    fn poll_depth_copy_complete(&mut self) -> bool {
        let Some(cache) = self.depth_fd.as_mut() else {
            return true;
        };
        let Some((bits, _gen)) = cache.pending else {
            return true;
        };
        let sync = bits as gl::types::GLsync;
        let status = unsafe { gl::ClientWaitSync(sync, 0, 0) };
        if status == gl::ALREADY_SIGNALED || status == gl::CONDITION_SATISFIED {
            unsafe { gl::DeleteSync(sync) };
            cache.pending = None;
            true
        } else if status == gl::WAIT_FAILED {
            // Failed fence must not pin the slot forever: treat as done so
            // the caller unpins (pixels may or may not have landed; the next
            // present heals via newest-wins).
            unsafe { gl::DeleteSync(sync) };
            cache.pending = None;
            warn!("depth FD fence wait failed; unpinning anyway");
            true
        } else {
            false
        }
    }

    fn drop_depth_imports(&mut self) {
        if let Some(cache) = self.depth_fd.take() {
            unsafe {
                if let Some((bits, _)) = cache.pending {
                    gl::DeleteSync(bits as gl::types::GLsync);
                }
                for s in &cache.slots {
                    if let Some(f) = DEPTH_EXT.delete_semaphores {
                        f(1, &s.sem);
                    }
                    gl::DeleteTextures(1, &s.tex);
                    if let Some(f) = DEPTH_EXT.delete_memory {
                        f(1, &s.mem);
                    }
                }
            }
        }
    }
}

impl GlData {
    /// Import both mirror slots (memory + depth texture + semaphore each).
    /// The FDs are borrowed, never closed or duplicated: the producer owns
    /// their lifetime (recycles only unpinned slots) and the SHM pin held
    /// by the caller covers this call.
    fn import_fd_slots(req: &DepthFdRequest, bpp: u64, internal: u32) -> Option<[DepthFdSlot; 2]> {
        // Driver allocation can pad past w*h*bpp (alignment). Import the
        // full allocation so glImportMemoryFdEXT never sees a truncated
        // object. The want > alloc rejection already ran in the caller;
        // alloc==0 means the producer predates the header field.
        let want = req.w as u64 * req.h as u64 * bpp;
        let size = want.max(req.alloc_bytes as u64);
        let import_mem = DEPTH_EXT.import_memory_fd?;
        let import_sem = DEPTH_EXT.import_semaphore_fd?;
        let create_mem = DEPTH_EXT.create_memory?;
        let tex_storage = DEPTH_EXT.texture_storage_mem_2d?;
        let gen_sem = DEPTH_EXT.gen_semaphores?;
        let mut slots = [DepthFdSlot::default(); 2];
        for i in 0..2 {
            let mut mem = 0;
            let mut tex = 0;
            let mut sem = 0;
            // Drain stale errors first so the step attribution below is
            // honest; bounded so a chatty driver can't stall us.
            for _ in 0..8 {
                if unsafe { gl::GetError() } == gl::NO_ERROR {
                    break;
                }
            }
            let step = unsafe {
                create_mem(1, &mut mem);
                if gl::GetError() != gl::NO_ERROR || mem == 0 {
                    "create-memory"
                } else {
                    import_mem(mem, size, HANDLE_TYPE_OPAQUE_FD, req.mem_fds[i]);
                    if gl::GetError() != gl::NO_ERROR {
                        "import-memory-fd"
                    } else {
                        gl::GenTextures(1, &mut tex);
                        gl::BindTexture(gl::TEXTURE_2D, tex);
                        tex_storage(tex, 1, internal, req.w as i32, req.h as i32, mem, 0);
                        gl::BindTexture(gl::TEXTURE_2D, 0);
                        if gl::GetError() != gl::NO_ERROR || tex == 0 {
                            "texture-storage-mem"
                        } else {
                            gen_sem(1, &mut sem);
                            if gl::GetError() != gl::NO_ERROR || sem == 0 {
                                "gen-semaphore"
                            } else {
                                import_sem(sem, HANDLE_TYPE_OPAQUE_FD, req.sem_fds[i]);
                                if gl::GetError() != gl::NO_ERROR {
                                    "import-semaphore-fd"
                                } else {
                                    ""
                                }
                            }
                        }
                    }
                }
            };
            if !step.is_empty() {
                warn!(
                    "depth FD import failed at {step} (slot {i}/2, epoch {}, {}x{} fmt {} memfd {} semfd {} size {})",
                    req.epoch, req.w, req.h, req.vk_format, req.mem_fds[i], req.sem_fds[i], size,
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

    /// Delete one partially- or fully-built slot. Zero handles are skipped
    /// (every delete entry point tolerates count>0 with valid handles only
    /// when nonzero; guard explicitly to never feed zeros).
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
                && let Some(f) = DEPTH_EXT.delete_memory
            {
                f(1, &s.mem);
            }
        }
    }
}

/// Texel footprint for the SHM Vulkan depth formats. D32S8 counts the
/// full combined texel (4 depth + 1 stencil + 3 pad = 8); the bound only
/// sizes the import, and the copy is depth-aspect-only.
fn depth_texel_bytes(vk_format: i32) -> Option<u64> {
    match vk_format {
        depth_feed::FORMAT_D32F => Some(4),
        depth_feed::FORMAT_D32S8 => Some(8),
        _ => None,
    }
}

/// GL internal format for the SHM Vulkan depth formats. The import
/// textures are sampled depth-only by the copy path (depth-aspect), so no
/// conversion is ever needed.
fn depth_gl_format(vk_format: i32) -> Option<u32> {
    match vk_format {
        depth_feed::FORMAT_D32F => Some(gl::DEPTH_COMPONENT32F),
        depth_feed::FORMAT_D32S8 => Some(gl::DEPTH32F_STENCIL8),
        _ => None,
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
    use super::{depth_gl_format, depth_texel_bytes};
    use crate::depth_feed::{FORMAT_D32F, FORMAT_D32S8};
    use gl as gl_enum;

    #[test]
    fn depth_format_maps_d32f_and_d32s8() {
        assert_eq!(depth_texel_bytes(FORMAT_D32F), Some(4));
        assert_eq!(depth_texel_bytes(FORMAT_D32S8), Some(8));
        assert_eq!(depth_texel_bytes(96), None);
        assert_eq!(
            depth_gl_format(FORMAT_D32F),
            Some(gl_enum::DEPTH_COMPONENT32F)
        );
        assert_eq!(
            depth_gl_format(FORMAT_D32S8),
            Some(gl_enum::DEPTH32F_STENCIL8)
        );
        assert_eq!(depth_gl_format(96), None);
    }
}
