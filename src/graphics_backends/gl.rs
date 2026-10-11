use super::GraphicsBackend;
use derive_more::Deref;
use glutin_glx_sys::{
    Success,
    glx::{self, Glx},
};
use libc::{dlerror, dlopen, dlsym};
use log::warn;
use openvr as vr;
use openxr as xr;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, LazyLock, Mutex, Once};

static GLX: LazyLock<Library> = LazyLock::new(|| Library::new(c"libGLX.so.0"));

pub struct GlData {
    session_data: Arc<SessionCreateInfo>,
    images: Vec<u32>,
    format: u32,
    read_fbo: u32,
    draw_fbo: u32,
    // Cached inverse-radial warp program for the XRIZER_WARP experiment.
    // Interior mutability so `copy_texture_to_swapchain(&self)` can lazily
    // compile it without changing the `GraphicsBackend` trait.
    warp: Mutex<GlWarpCache>,
}

#[derive(Default)]
struct GlWarpCache {
    prog: Option<GlWarpProg>,
    failed: bool,
}

struct GlWarpProg {
    program: gl::types::GLuint,
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    sampler: gl::types::GLuint,
}

/// Snapshot of the app's GL state touched by the warp pass, so Submit never
/// visibly disturbs the app's own context state.
struct SavedGlState {
    program: i32,
    vao: i32,
    array_buf: i32,
    active: i32,
    tex2d: i32,
    sampler: i32,
    viewport: [i32; 4],
    scissor_box: [i32; 4],
    read_fbo: i32,
    draw_fbo: i32,
    blend: u8,
    depth: u8,
    stencil: u8,
    scissor: u8,
    cull: u8,
}

impl SavedGlState {
    unsafe fn save() -> Self {
        let mut s = Self {
            program: 0,
            vao: 0,
            array_buf: 0,
            active: 0,
            tex2d: 0,
            sampler: 0,
            viewport: [0; 4],
            scissor_box: [0; 4],
            read_fbo: 0,
            draw_fbo: 0,
            blend: 0,
            depth: 0,
            stencil: 0,
            scissor: 0,
            cull: 0,
        };
        gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut s.program);
        gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut s.vao);
        gl::GetIntegerv(gl::ARRAY_BUFFER_BINDING, &mut s.array_buf);
        gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut s.active);
        // Unit-0 bindings must be read with unit 0 active; the app may
        // have left any other unit selected.
        gl::ActiveTexture(gl::TEXTURE0);
        gl::GetIntegerv(gl::TEXTURE_BINDING_2D, &mut s.tex2d);
        gl::GetIntegerv(gl::SAMPLER_BINDING, &mut s.sampler);
        gl::GetIntegerv(gl::VIEWPORT, s.viewport.as_mut_ptr());
        gl::GetIntegerv(gl::SCISSOR_BOX, s.scissor_box.as_mut_ptr());
        gl::GetIntegerv(gl::READ_FRAMEBUFFER_BINDING, &mut s.read_fbo);
        gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut s.draw_fbo);
        s.blend = gl::IsEnabled(gl::BLEND);
        s.depth = gl::IsEnabled(gl::DEPTH_TEST);
        s.stencil = gl::IsEnabled(gl::STENCIL_TEST);
        s.scissor = gl::IsEnabled(gl::SCISSOR_TEST);
        s.cull = gl::IsEnabled(gl::CULL_FACE);
        s
    }

    unsafe fn restore(&self) {
        gl::BindSampler(0, self.sampler as u32);
        gl::BindTexture(gl::TEXTURE_2D, self.tex2d as u32);
        gl::ActiveTexture(self.active as u32);
        gl::BindVertexArray(self.vao as u32);
        gl::BindBuffer(gl::ARRAY_BUFFER, self.array_buf as u32);
        gl::UseProgram(self.program as u32);
        gl::Viewport(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        if self.scissor != 0 {
            gl::Enable(gl::SCISSOR_TEST);
        }
        gl::Scissor(
            self.scissor_box[0],
            self.scissor_box[1],
            self.scissor_box[2],
            self.scissor_box[3],
        );
        if self.blend != 0 {
            gl::Enable(gl::BLEND);
        }
        if self.depth != 0 {
            gl::Enable(gl::DEPTH_TEST);
        }
        if self.stencil != 0 {
            gl::Enable(gl::STENCIL_TEST);
        }
        if self.cull != 0 {
            gl::Enable(gl::CULL_FACE);
        }
        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo as u32);
        gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_fbo as u32);
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

        Some(GlData {
            session_data: Arc::new(SessionCreateInfo(session_info)),
            images: Default::default(),
            format: 0,
            read_fbo: fbos[0],
            draw_fbo: fbos[1],
            warp: Mutex::default(),
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
        // XRIZER_WARP experiment: inverse-radial post-process instead of the
        // straight CopyImageSubData/BlitFramebuffer below. k == 0 keeps the
        // original paths.
        let warp_k = crate::system::warp_strength();
        if warp_k != 0.0 {
            static LOGGED_GL_WARP: std::sync::Once = std::sync::Once::new();
            LOGGED_GL_WARP.call_once(|| {
                log::info!(
                    "xrizer-debug: warp active on OpenGL backend k={warp_k:.4} (k>0 pincushion, k<0 barrel)"
                );
            });
            if let Some(extent) =
                self.copy_texture_warped(eye, texture, bounds, image_index, warp_k)
            {
                return extent;
            }
            // Program compile/link failure falls through to the straight copy.
        }

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
}

// Inverse-radial warp (XRIZER_WARP) for the OpenGL backend.
//
// The app's GL context is current on this thread during Submit, so this is
// all immediate-mode GL: attach the swapchain eye layer to our draw FBO and
// render a fullscreen quad sampling the game texture with
// `r' = r*(1 + k*r^2)` around the FOV-derived center. All touched GL state
// (program, VAO, textures, viewport, framebuffers, blend/depth/scissor/cull)
// is saved and restored so the app never observes us.
impl GlData {
    const WARP_VERT_SRC: &'static str = r#"#version 330 core
layout(location = 0) in vec2 pos;
out vec2 baseUV;
uniform vec2 uRange; // min,max source U of the submitted sub-rect
uniform vec2 vRange; // min,max source V of the submitted sub-rect
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    vec2 px = pos * 0.5 + 0.5;
    baseUV = vec2(mix(uRange.x, uRange.y, px.x), mix(vRange.x, vRange.y, px.y));
}
"#;

    const WARP_FRAG_SRC: &'static str = r#"#version 330 core
in vec2 baseUV;
out vec4 color;
uniform sampler2D src;
uniform vec2 center; // optical center in baseUV space
uniform float k;     // >0 pincushion, <0 barrel, 0 off
uniform float aspect; // extent.width / extent.height
void main() {
    vec2 p = vec2((baseUV.x - center.x) * aspect, baseUV.y - center.y);
    float r2 = dot(p, p);
    vec2 pw = p * (1.0 + k * r2);
    vec2 uv = vec2(pw.x / aspect + center.x, pw.y + center.y);
    color = texture(src, uv);
}
"#;

    fn compile_warp_program() -> Option<GlWarpProg> {
        unsafe fn info_log(line: impl FnOnce(*mut i32, *mut c_char)) -> String {
            let mut len = 0;
            let mut buf = [0 as c_char; 1024];
            line(&mut len, buf.as_mut_ptr());
            let len = (len as usize).min(buf.len() - 1);
            String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr().cast(), len))
                .into_owned()
        }

        unsafe fn compile_stage(ty: gl::types::GLenum, src: &str) -> Option<gl::types::GLuint> {
            let sh = gl::CreateShader(ty);
            let csrc = CString::new(src).ok()?;
            gl::ShaderSource(sh, 1, &csrc.as_ptr(), std::ptr::null());
            gl::CompileShader(sh);
            let mut ok = 0;
            gl::GetShaderiv(sh, gl::COMPILE_STATUS, &mut ok);
            if ok == 0 {
                let log = info_log(|len, buf| {
                    gl::GetShaderInfoLog(sh, 1024, len, buf);
                });
                log::warn!("xrizer-debug: warp GL shader compile failed: {log}");
                gl::DeleteShader(sh);
                return None;
            }
            Some(sh)
        }

        unsafe {
            let vs = compile_stage(gl::VERTEX_SHADER, Self::WARP_VERT_SRC)?;
            let fs = compile_stage(gl::FRAGMENT_SHADER, Self::WARP_FRAG_SRC)?;
            let program = gl::CreateProgram();
            gl::AttachShader(program, vs);
            gl::AttachShader(program, fs);
            gl::LinkProgram(program);
            gl::DeleteShader(vs);
            gl::DeleteShader(fs);
            let mut ok = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut ok);
            if ok == 0 {
                let log = info_log(|len, buf| {
                    gl::GetProgramInfoLog(program, 1024, len, buf);
                });
                log::warn!("xrizer-debug: warp GL program link failed: {log}");
                gl::DeleteProgram(program);
                return None;
            }

            let mut vao = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            // Fullscreen TRIANGLE_STRIP quad in NDC.
            let verts: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
            gl::BufferData(
                gl::ARRAY_BUFFER,
                std::mem::size_of_val(&verts) as isize,
                verts.as_ptr().cast(),
                gl::STATIC_DRAW,
            );
            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
            gl::EnableVertexAttribArray(0);
            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            let mut sampler = 0;
            gl::GenSamplers(1, &mut sampler);
            gl::SamplerParameteri(sampler, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::SamplerParameteri(sampler, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl::SamplerParameteri(sampler, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::SamplerParameteri(sampler, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);

            Some(GlWarpProg {
                program,
                vao,
                vbo,
                sampler,
            })
        }
    }

    fn uniform(program: gl::types::GLuint, name: &CStr) -> gl::types::GLint {
        unsafe { gl::GetUniformLocation(program, name.as_ptr().cast()) }
    }

    /// Warp submit path; `None` means "fall back to the straight copy"
    /// (program failed to compile/link). Caller guarantees `k != 0`.
    fn copy_texture_warped(
        &self,
        eye: vr::EVREye,
        texture: gl::types::GLuint,
        bounds: vr::VRTextureBounds_t,
        image_index: usize,
        k: f32,
    ) -> Option<xr::Extent2Di> {
        // Lazily compile the program; remember failure so we pay for the
        // attempt only once and fall back quietly afterwards.
        {
            let mut cache = self.warp.lock().unwrap();
            if cache.prog.is_none() && !cache.failed {
                match Self::compile_warp_program() {
                    Some(prog) => cache.prog = Some(prog),
                    None => {
                        cache.failed = true;
                        return None;
                    }
                }
            } else if cache.prog.is_none() {
                return None;
            }
        }
        let cache = self.warp.lock().unwrap();
        let prog = cache.prog.as_ref()?;

        let xr::Rect2Di { extent, .. } = texture_rect_from_bounds(texture, bounds);
        if extent.width == 0 || extent.height == 0 {
            return None;
        }
        let (cx, cy) = crate::system::warp_center_for_eye_gl((eye as usize) == 1, bounds);
        let aspect = extent.width as f32 / extent.height as f32;
        // Ranges in min,max order so framebuffer-bottom maps to the minimum-V
        // row, exactly like CopyImageSubData does (center math above matches).
        let (u0, u1) = (bounds.uMin.min(bounds.uMax), bounds.uMin.max(bounds.uMax));
        let (v0, v1) = (bounds.vMin.min(bounds.vMax), bounds.vMin.max(bounds.vMax));
        log::trace!("xrizer-debug: warp GL submit eye={eye:?} k={k:.4} cx={cx:.4} cy={cy:.4}");

        let swapchain_texture = self.images[image_index];

        unsafe {
            let saved = SavedGlState::save();

            let restore_and_out = |saved: &SavedGlState| {
                saved.restore();
                None
            };

            // ---- render warp quad into the swapchain eye layer ----
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_fbo);
            gl::FramebufferTextureLayer(
                gl::DRAW_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                swapchain_texture,
                0,
                eye as i32,
            );
            gl::DrawBuffer(gl::COLOR_ATTACHMENT0);
            if gl::CheckFramebufferStatus(gl::DRAW_FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                log::warn!(
                    "xrizer-debug: warp GL framebuffer incomplete, falling back to straight copy"
                );
                return restore_and_out(&saved);
            }
            gl::Viewport(0, 0, extent.width, extent.height);

            gl::Disable(gl::BLEND);
            gl::Disable(gl::DEPTH_TEST);
            gl::Disable(gl::STENCIL_TEST);
            gl::Disable(gl::SCISSOR_TEST);
            gl::Disable(gl::CULL_FACE);

            gl::UseProgram(prog.program);
            gl::BindVertexArray(prog.vao);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::BindSampler(0, prog.sampler);

            let set2 = |name: &CStr, x: f32, y: f32| {
                let loc = Self::uniform(prog.program, name);
                if loc >= 0 {
                    gl::Uniform2f(loc, x, y);
                }
            };
            let set1 = |name: &CStr, x: f32| {
                let loc = Self::uniform(prog.program, name);
                if loc >= 0 {
                    gl::Uniform1f(loc, x);
                }
            };
            set2(c"uRange", u0, u1);
            set2(c"vRange", v0, v1);
            set2(c"center", cx, cy);
            set1(c"k", k);
            set1(c"aspect", aspect);
            let src_loc = Self::uniform(prog.program, c"src");
            if src_loc >= 0 {
                gl::Uniform1i(src_loc, 0);
            }

            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

            // ---- restore app GL state ----
            saved.restore();
        }

        Some(extent)
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
