mod gl;
mod vulkan;

use derive_more::{From, TryInto};
pub use gl::GlData;
use openvr as vr;
use openxr as xr;
pub use vulkan::VulkanData;

pub trait GraphicsBackend: Into<SupportedBackend> {
    type Api: xr::Graphics + 'static;
    type OpenVrTexture: Copy;
    type NiceFormat: std::fmt::Debug;

    fn to_nice_format(format: <Self::Api as xr::Graphics>::Format) -> Self::NiceFormat;

    fn session_create_info(&self) -> <Self::Api as xr::Graphics>::SessionCreateInfo;

    /// Returns None if the texture is invalid.
    fn get_texture(texture: &vr::Texture_t) -> Option<Self::OpenVrTexture>;

    fn swapchain_info_for_texture(
        &self,
        texture: Self::OpenVrTexture,
        bounds: vr::VRTextureBounds_t,
        color_space: vr::EColorSpace,
    ) -> xr::SwapchainCreateInfo<Self::Api>;

    fn store_swapchain_images(
        &mut self,
        images: Vec<<Self::Api as xr::Graphics>::SwapchainImage>,
        format: <Self::Api as xr::Graphics>::Format,
    );

    fn copy_texture_to_swapchain(
        &self,
        eye: vr::EVREye,
        texture: Self::OpenVrTexture,
        color_space: vr::EColorSpace,
        bounds: vr::VRTextureBounds_t,
        image_index: usize,
        submit_flags: vr::EVRSubmitFlags,
    ) -> xr::Extent2Di;

    fn copy_overlay_to_swapchain(
        &mut self,
        texture: Self::OpenVrTexture,
        bounds: vr::VRTextureBounds_t,
        image_index: usize,
    ) -> xr::Extent2Di;

    /// Depth handoff (GL session only). Defaults keep other backends
    /// compiling untouched and color-only.
    fn depth_swapchain_create_info(
        &self,
        color_info: &xr::SwapchainCreateInfo<Self::Api>,
        depth_format: <Self::Api as xr::Graphics>::Format,
    ) -> Option<xr::SwapchainCreateInfo<Self::Api>> {
        let _ = (color_info, depth_format);
        None
    }

    fn store_depth_swapchain_images(
        &mut self,
        images: Vec<<Self::Api as xr::Graphics>::SwapchainImage>,
        format: <Self::Api as xr::Graphics>::Format,
    ) {
        let _ = (images, format);
    }

    /// Whether this backend can carry depth at all. Used to skip SHM work.
    fn supports_depth() -> bool {
        false
    }

    /// GL swapchain format for a SHM Vulkan depth format. Checked against
    /// the session's enumerated formats; depth stays off when absent.
    fn depth_format_for_header(vk_format: i32) -> Option<<Self::Api as xr::Graphics>::Format> {
        let _ = vk_format;
        None
    }

    /// Whether this backend can import opaque-FD depth (GL extension probe).
    fn depth_fd_import_viable(&self) -> bool {
        false
    }

    /// Ensure FD imports for (epoch, dims, format); GPU-copy both halves
    /// of `generation` into `depth_image_index`. The caller holds the
    /// slot's SHM pin across this call; the backend queues a GPU wait on
    /// the slot semaphore before copying and fences its own completion.
    /// Default: unsupported (keeps other backends compiling untouched).
    fn import_and_copy_depth_fd(&mut self, req: &DepthFdRequest) -> DepthFdResult {
        let _ = req;
        DepthFdResult { copied: false }
    }

    /// True once a previous [`Self::import_and_copy_depth_fd`] fence has
    /// signaled (zero-timeout poll). The caller unpins on true.
    fn poll_depth_copy_complete(&mut self) -> bool {
        false
    }

    /// Delete FD import GL objects (reimport path / teardown).
    fn drop_depth_imports(&mut self) {}
}

/// One FD-mode frame for [`GraphicsBackend::import_and_copy_depth_fd`].
/// Two slots (`generation % 2`); FDs are borrowed numbers, never closed.
pub struct DepthFdRequest {
    pub epoch: u32,
    pub w: u32,
    pub h: u32,
    pub vk_format: i32,
    pub mem_fds: [i32; 2],
    pub sem_fds: [i32; 2],
    pub generation: u64,
    pub depth_image_index: usize,
    pub eye_w: u32,
    pub eye_h: u32,
    /// Driver-reported allocation size from the SHM header. The GL import
    /// covers the full allocation; `want > alloc` (non-zero) rejects the
    /// setup before any GL call. 0 = unknown, use `w*h*bpp`.
    pub alloc_bytes: u32,
}

/// Outcome of [`GraphicsBackend::import_and_copy_depth_fd`].
pub struct DepthFdResult {
    /// Both halves copied (queued) this call; end_frame may chain depth.
    pub copied: bool,
}

#[derive(macros::Backends, TryInto, From)]
#[try_into(owned, ref)]
#[allow(clippy::large_enum_variant)]
pub enum SupportedBackend {
    Vulkan(VulkanData),
    OpenGL(GlData),
    #[cfg(test)]
    Fake(crate::compositor::FakeGraphicsData),
}

impl<B> GraphicsEnum<B> for SupportedBackend
where
    B: GraphicsBackend + TryFrom<Self>,
{
    type Inner = B;
}

#[allow(clippy::single_component_path_imports)]
pub(crate) use supported_apis_enum;
#[allow(clippy::single_component_path_imports)]
pub(crate) use supported_backends_enum;

// These traits are used to commit some type crimes, see macros::any_graphics
// None of them should be manually implemented
pub trait GraphicsEnum<G>: Sized {
    type Inner: TryFrom<Self>;
}

pub trait WithAnyGraphicsParams {
    type Args;
    type Ret;
}

pub trait WithAnyGraphics<G>: WithAnyGraphicsParams {
    type GraphicsEnum: GraphicsEnum<G>;
    fn with_any_graphics(
        inner: &<Self::GraphicsEnum as GraphicsEnum<G>>::Inner,
        args: Self::Args,
    ) -> Self::Ret;
}

pub trait WithAnyGraphicsMut<G>: WithAnyGraphicsParams {
    type GraphicsEnum: GraphicsEnum<G>;
    fn with_any_graphics(
        inner: &mut <Self::GraphicsEnum as GraphicsEnum<G>>::Inner,
        args: Self::Args,
    ) -> Self::Ret;
}

pub trait WithAnyGraphicsOwned<G>: WithAnyGraphicsParams {
    type GraphicsEnum: GraphicsEnum<G>;
    fn with_any_graphics(
        inner: <Self::GraphicsEnum as GraphicsEnum<G>>::Inner,
        args: Self::Args,
    ) -> Self::Ret;
}

impl SupportedBackend {
    pub fn is_texture_type_supported(texture_type: vr::ETextureType) -> bool {
        match texture_type {
            vr::ETextureType::Vulkan | vr::ETextureType::OpenGL => true,
            #[cfg(test)]
            vr::ETextureType::Reserved => true,
            _ => false,
        }
    }

    pub fn new(texture: &vr::Texture_t, _bounds: vr::VRTextureBounds_t) -> Option<Self> {
        match texture.eType {
            vr::ETextureType::Vulkan => {
                let vk_texture = unsafe { &*(texture.handle as *const vr::VRVulkanTextureData_t) };
                Some(Self::Vulkan(VulkanData::new(vk_texture)))
            }
            vr::ETextureType::OpenGL => GlData::new().map(Self::OpenGL),
            #[cfg(test)]
            vr::ETextureType::Reserved => Some(Self::Fake(
                crate::compositor::FakeGraphicsData::new(texture),
            )),
            other => panic!("Unsupported texture type: {other:?}"),
        }
    }
}
