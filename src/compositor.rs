use crate::{
    AtomicF64,
    clientcore::{Injected, Injector},
    depth_feed::{
        AgeStats, DepthFeed, FORMAT_D32F, FORMAT_D32S8, FencePoll, MAX_DEPTH, MIN_DEPTH, ReadSkip,
        depth_master_enabled, is_same_generation, now_mono_ns, poll_fence_fd, validate_near_far,
    },
    graphics_backends::{
        DepthFdRequest, GraphicsBackend, SupportedBackend, supported_backends_enum,
    },
    input::Input,
    openxr_data::{self, FrameStream, OpenXrData, SessionCreateInfo, SessionData},
    overlay::OverlayMan,
    system::System,
    tracy_span,
};

use log::{debug, info, trace, warn};
use openvr as vr;
use openxr as xr;
use std::mem::offset_of;
use std::sync::{
    Arc, Mutex, Once,
    atomic::{AtomicU32, Ordering},
};
use std::time::{Duration, Instant};
use std::{ffi::c_char, ops::Deref};

#[derive(Default)]
pub struct CompositorSessionData(Mutex<Option<DynFrameController>>);

#[derive(macros::InterfaceImpl)]
#[interface = "IVRCompositor"]
#[versions(
    029, 028, 027, 026, 022, 021, 020, 019, 018, 016, 015, 014, 012, 011, 009
)]
pub struct Compositor {
    vtables: Vtables,
    openxr: Arc<OpenXrData<Self>>,
    input: Injected<Input<Self>>,
    system: Injected<System>,
    /// Stores the backend data in between session restarts.
    tmp_backend: Mutex<Option<AnyTempBackendData>>,
    overlays: Injected<OverlayMan>,
    metrics: FrameMetrics,
    timing_mode: Mutex<vr::EVRCompositorTimingMode>,
    frame_state: Mutex<FrameState>,
    focused: Once,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FrameState {
    Waited,
    Begun,
    Submitted,
}

impl FrameState {
    fn advance_to(&mut self, new: Self) -> bool {
        let old = *self;
        let allowed: bool = match old {
            Self::Waited => new == Self::Begun,
            Self::Begun => new != Self::Begun,
            Self::Submitted => new == Self::Waited,
        };
        if allowed {
            *self = new;
            trace!("advanced frame state from {old:?} to {new:?}");
        } else {
            warn!("Tried to advance frame state from {old:?} to {new:?}, which is not allowed");
        }
        allowed
    }
}

struct FrameMetrics {
    system_start: Instant,
    index: AtomicU32,
    time: AtomicF64,
}

struct TempBackendData<G: GraphicsBackend> {
    backend: G,
    swapchain_create_info: Option<xr::SwapchainCreateInfo<G::Api>>,
}
supported_backends_enum!(enum AnyTempBackendData: TempBackendData);

impl Compositor {
    pub fn new(openxr: Arc<OpenXrData<Self>>, injector: &Injector) -> Self {
        Self {
            vtables: Default::default(),
            openxr,
            input: injector.inject(),
            system: injector.inject(),
            tmp_backend: Mutex::default(),
            overlays: injector.inject(),
            metrics: FrameMetrics {
                system_start: Instant::now(),
                index: 0.into(),
                time: 0.0.into(),
            },
            timing_mode: vr::EVRCompositorTimingMode::Implicit.into(),
            frame_state: FrameState::Submitted.into(),
            focused: Once::new(),
        }
    }

    fn maybe_wait_frame(&self, session_data: &SessionData) {
        tracy_span!();
        let mut frame_lock = { session_data.comp_data.0.lock().unwrap() };
        self.frame_state
            .lock()
            .unwrap()
            .advance_to(FrameState::Waited);
        let Some(ctrl) = frame_lock.as_mut() else {
            debug!("no frame controller - not starting frame");
            return;
        };

        #[macros::any_graphics(DynFrameController)]
        fn wait_frame<G: GraphicsBackend + 'static>(
            ctrl: &mut FrameController<G>,
        ) -> (xr::Time, i64) {
            ctrl.wait_frame()
        }

        let (display_time, display_period) = ctrl.with_any_graphics_mut::<wait_frame>(());
        self.openxr.display_time.set(display_time);
        self.openxr
            .display_period_nanos
            .store(display_period, Ordering::Relaxed);
    }

    fn maybe_begin_frame(&self, session_data: &SessionData) {
        tracy_span!();
        let mut frame_lock = { session_data.comp_data.0.lock().unwrap() };
        if !self
            .frame_state
            .lock()
            .unwrap()
            .advance_to(FrameState::Begun)
        {
            debug!("not starting frame - already begun or submitted");
            return;
        };
        let Some(ctrl) = frame_lock.as_mut() else {
            debug!("no frame controller - not starting frame");
            return;
        };

        #[macros::any_graphics(DynFrameController)]
        fn begin_frame<G: GraphicsBackend + 'static>(ctrl: &mut FrameController<G>) {
            ctrl.begin_frame()
        }

        ctrl.with_any_graphics_mut::<begin_frame>(());
    }

    pub fn initialize_real_session(
        &self,
        texture: &vr::Texture_t,
        bounds: vr::VRTextureBounds_t,
    ) -> Result<(), vr::EVRCompositorError> {
        let backend =
            SupportedBackend::new(texture, bounds).ok_or(vr::EVRCompositorError::InvalidTexture)?;

        #[macros::any_graphics(SupportedBackend)]
        fn swapchain_info<G: GraphicsBackend>(
            backend: G,
            texture: &vr::Texture_t,
            bounds: vr::VRTextureBounds_t,
        ) -> Result<AnyTempBackendData, vr::EVRCompositorError>
        where
            AnyTempBackendData: From<TempBackendData<G>>,
        {
            let b_texture =
                G::get_texture(texture).ok_or(vr::EVRCompositorError::InvalidTexture)?;
            let info = backend.swapchain_info_for_texture(b_texture, bounds, texture.eColorSpace);
            Ok(TempBackendData {
                backend,
                swapchain_create_info: Some(info),
            }
            .into())
        }

        let tmp_backend = backend
            .with_any_graphics_owned::<swapchain_info>((texture, bounds))
            .inspect_err(|_| debug!("received invalid texture handle, not restarting session"))?;
        *self.tmp_backend.lock().unwrap() = Some(tmp_backend);

        info!("Creating real backend for texture type {:?}", texture.eType);
        self.openxr.restart_session();
        Ok(())
    }
}

fn fill_vk_extensions_buffer(extensions: String, buffer: *mut c_char, buffer_size: u32) -> u32 {
    let bytes = unsafe {
        std::slice::from_raw_parts(extensions.as_ptr() as *const c_char, extensions.len())
    };

    if !buffer.is_null() && buffer_size as usize > bytes.len() {
        let buffer = unsafe { std::slice::from_raw_parts_mut(buffer, buffer_size as usize) };
        buffer[0..bytes.len()].copy_from_slice(bytes);
        buffer[bytes.len()] = 0;
    }

    bytes.len() as u32 + 1
}

impl openxr_data::Compositor for Compositor {
    fn get_session_create_info(&self, data: CompositorSessionData) -> SessionCreateInfo {
        #[macros::any_graphics(AnyTempBackendData)]
        fn info<G: GraphicsBackend>(data: &TempBackendData<G>) -> SessionCreateInfo
        where
            SessionCreateInfo: From<openxr_data::CreateInfo<G::Api>>,
        {
            SessionCreateInfo::from_info::<G::Api>(data.backend.session_create_info())
        }

        self.tmp_backend
            .lock()
            .unwrap()
            .get_or_insert_with(|| {
                #[macros::any_graphics(DynFrameController)]
                fn take<G: GraphicsBackend>(mut ctrl: FrameController<G>) -> AnyTempBackendData
                where
                    AnyTempBackendData: From<TempBackendData<G>>,
                {
                    // Release a held SHM pin before the controller (and its
                    // feed) is dropped: otherwise the slot stays pinned
                    // forever (silent permanent half-rate until the producer
                    // rebuilds the segment).
                    ctrl.drop_depth_state();
                    TempBackendData {
                        backend: ctrl.backend,
                        swapchain_create_info: ctrl.swapchain_data.map(|d| d.info),
                    }
                    .into()
                }
                data.0
                    .lock()
                    .unwrap()
                    .take()
                    .expect("one of tmp backend or frame controller should be setup")
                    .with_any_graphics_owned::<take>(())
            })
            .with_any_graphics::<info>(())
    }

    fn post_session_restart(
        &self,
        session_data: &SessionData,
        waiter: xr::FrameWaiter,
        stream: FrameStream,
    ) {
        // This function is called while a write lock is called on the session, and as such should
        // not use self.openxr.session_data.get().

        let backend_data = self
            .tmp_backend
            .lock()
            .unwrap()
            .take()
            .expect("tmp backend should be set before post_session_restart is called");

        #[macros::any_graphics(AnyTempBackendData)]
        fn new_frame_controller<G: GraphicsBackend + 'static>(
            data: TempBackendData<G>,
            session_data: &SessionData,
            waiter: xr::FrameWaiter,
            stream: FrameStream,
        ) -> DynFrameController
        where
            for<'a> &'a openxr_data::GraphicalSession:
                TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
            FrameStream: TryInto<xr::FrameStream<G::Api>>,
            DynFrameController: From<FrameController<G>>,
            <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
        {
            FrameController::new(
                session_data,
                waiter,
                stream.try_into().unwrap_or_else(|_| unreachable!()),
                data.backend,
                data.swapchain_create_info,
            )
            .into()
        }

        *session_data.comp_data.0.lock().unwrap() = Some(
            backend_data.with_any_graphics_owned::<new_frame_controller>((
                session_data,
                waiter,
                stream,
            )),
        );

        let old_state = std::mem::replace(
            &mut *self.frame_state.lock().unwrap(),
            FrameState::Submitted,
        );

        trace!("returning to {old_state:?} frame state");
        match old_state {
            FrameState::Submitted => {}
            FrameState::Waited => self.maybe_wait_frame(session_data),
            FrameState::Begun => {
                self.maybe_wait_frame(session_data);
                self.maybe_begin_frame(session_data);
            }
        }
    }
}

#[allow(non_snake_case)]
impl vr::IVRCompositor029_Interface for Compositor {
    fn GetPosesForFrame(
        &self,
        _unPosePredictionID: u32,
        _pPoseArray: *mut vr::TrackedDevicePose_t,
        _unPoseArrayCount: u32,
    ) -> vr::EVRCompositorError {
        todo!()
    }
    fn GetLastPosePredictionIDs(
        &self,
        _pRenderPosePredictionID: *mut u32,
        _pGamePosePredictionID: *mut u32,
    ) -> vr::EVRCompositorError {
        crate::warn_unimplemented!("GetLastPosePredictionIDs");
        vr::EVRCompositorError::None
    }
    fn GetCompositorBenchmarkResults(
        &self,
        _pBenchmarkResults: *mut vr::Compositor_BenchmarkResults,
        _nSizeOfBenchmarkResults: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetCompositorBenchmarkResults");
        false
    }
    fn ClearStageOverride(&self) {}
    fn SetStageOverride_Async(
        &self,
        _pchRenderModelPath: *const std::ffi::c_char,
        _pTransform: *const vr::HmdMatrix34_t,
        _pRenderSettings: *const vr::Compositor_StageRenderSettings,
        _nSizeOfRenderSettings: u32,
    ) -> vr::EVRCompositorError {
        crate::warn_unimplemented!("SetStageOverride_Async");
        vr::EVRCompositorError::None
    }
    fn IsCurrentSceneFocusAppLoading(&self) -> bool {
        false
    }
    fn IsMotionSmoothingSupported(&self) -> bool {
        todo!()
    }
    fn IsMotionSmoothingEnabled(&self) -> bool {
        todo!()
    }
    fn SubmitExplicitTimingData(&self) -> vr::EVRCompositorError {
        if *self.timing_mode.lock().unwrap() == vr::EVRCompositorTimingMode::Implicit {
            return vr::EVRCompositorError::RequestFailed;
        }

        if !self.focused.is_completed() {
            // SteamVR doesn't seem to return an error in this case...
            return vr::EVRCompositorError::None;
        }

        let session_data = self.openxr.session_data.get();
        self.maybe_begin_frame(&session_data);
        vr::EVRCompositorError::None
    }
    fn SetExplicitTimingMode(&self, timing_mode: vr::EVRCompositorTimingMode) {
        debug!("Setting timing mode to {timing_mode:?}");
        *self.timing_mode.lock().unwrap() = timing_mode;
    }

    fn GetVulkanDeviceExtensionsRequired(
        &self,
        _physical_device: *mut vr::VkPhysicalDevice_T,
        buffer: *mut std::ffi::c_char,
        buffer_size: u32,
    ) -> u32 {
        let exts = self
            .openxr
            .instance
            .vulkan_legacy_device_extensions(self.openxr.system_id)
            .unwrap();
        log::debug!("required device extensions: {exts}");
        fill_vk_extensions_buffer(exts, buffer, buffer_size)
    }

    fn GetVulkanInstanceExtensionsRequired(
        &self,
        buffer: *mut std::ffi::c_char,
        buffer_size: u32,
    ) -> u32 {
        let exts = self
            .openxr
            .instance
            .vulkan_legacy_instance_extensions(self.openxr.system_id)
            .unwrap();
        log::debug!("required instance extensions: {exts}");
        fill_vk_extensions_buffer(exts, buffer, buffer_size)
    }

    fn UnlockGLSharedTextureForAccess(&self, _glSharedTextureHandle: vr::glSharedTextureHandle_t) {
        todo!()
    }
    fn LockGLSharedTextureForAccess(&self, _glSharedTextureHandle: vr::glSharedTextureHandle_t) {
        todo!()
    }
    fn ReleaseSharedGLTexture(
        &self,
        _glTextureId: vr::glUInt_t,
        _glSharedTextureHandle: vr::glSharedTextureHandle_t,
    ) -> bool {
        todo!()
    }
    fn GetMirrorTextureGL(
        &self,
        _eEye: vr::EVREye,
        _pglTextureId: *mut vr::glUInt_t,
        _pglSharedTextureHandle: *mut vr::glSharedTextureHandle_t,
    ) -> vr::EVRCompositorError {
        todo!()
    }
    fn ReleaseMirrorTextureD3D11(&self, _pD3D11ShaderResourceView: *mut std::ffi::c_void) {
        todo!()
    }
    fn GetMirrorTextureD3D11(
        &self,
        _eEye: vr::EVREye,
        _pD3D11DeviceOrResource: *mut std::ffi::c_void,
        _ppD3D11ShaderResourceView: *mut *mut std::ffi::c_void,
    ) -> vr::EVRCompositorError {
        crate::warn_unimplemented!("GetMirrorTextureD3D11");
        vr::EVRCompositorError::IncompatibleVersion
    }
    fn SuspendRendering(&self, bSuspend: bool) {
        #[macros::any_graphics(DynFrameController)]
        fn set_suspend_render<G: GraphicsBackend + 'static>(
            ctrl: &mut FrameController<G>,
            app_suspend_render: bool,
        ) {
            ctrl.app_suspend_render = app_suspend_render;
        }

        self.openxr
            .session_data
            .get()
            .comp_data
            .0
            .lock()
            .unwrap()
            .iter_mut()
            .for_each(|ctrl| ctrl.with_any_graphics_mut::<set_suspend_render>(bSuspend));
    }
    fn ForceReconnectProcess(&self) {
        todo!()
    }
    fn ForceInterleavedReprojectionOn(&self, _: bool) {
        crate::warn_unimplemented!("ForceInterleavedReprojectionOn");
    }
    fn ShouldAppRenderWithLowResources(&self) -> bool {
        // TODO
        false
    }
    fn CompositorDumpImages(&self) {
        todo!()
    }
    fn IsMirrorWindowVisible(&self) -> bool {
        todo!()
    }
    fn HideMirrorWindow(&self) {
        todo!()
    }
    fn ShowMirrorWindow(&self) {
        todo!()
    }
    fn CanRenderScene(&self) -> bool {
        true
    }
    fn GetLastFrameRenderer(&self) -> u32 {
        0
    }
    fn GetCurrentSceneFocusProcess(&self) -> u32 {
        todo!()
    }
    fn IsFullscreen(&self) -> bool {
        true
    }
    fn CompositorQuit(&self) {
        todo!()
    }
    fn CompositorGoToBack(&self) {
        crate::warn_unimplemented!("CompositorGoToBack");
    }
    fn CompositorBringToFront(&self) {
        crate::warn_unimplemented!("CompositorBringToFront");
    }
    fn ClearSkyboxOverride(&self) {
        if let Some(overlays) = self.overlays.get() {
            overlays.clear_skybox();
        }
    }
    fn SetSkyboxOverride(
        &self,
        pTextures: *const vr::Texture_t,
        unTextureCount: u32,
    ) -> vr::EVRCompositorError {
        let overlays = self
            .overlays
            .force(|injector| OverlayMan::new(self.openxr.clone(), injector));
        if pTextures.is_null() {
            return vr::EVRCompositorError::RequestFailed;
        }
        match unTextureCount {
            1..=2 => {
                if !self
                    .openxr
                    .enabled_extensions
                    .khr_composition_layer_equirect2
                {
                    log::info!("Could not set skybox: khr_composition_layer_equirect2 unsupported");
                    return vr::EVRCompositorError::None;
                }
                log::debug!("Setting new equirect skybox");
            }
            6 => {
                log::debug!("Setting new box skybox");
            }
            _ => {
                log::warn!("Invalid number of skybox textures: {unTextureCount}");
                return vr::EVRCompositorError::RequestFailed;
            }
        }

        let textures = unsafe { std::slice::from_raw_parts(pTextures, unTextureCount as _) };
        if let Err(e) = overlays.set_skybox(textures) {
            e
        } else {
            vr::EVRCompositorError::None
        }
    }
    fn GetCurrentGridAlpha(&self) -> f32 {
        0.0
    }
    fn FadeGrid(&self, _fSeconds: f32, bFadeGridIn: bool) {
        #[macros::any_graphics(DynFrameController)]
        fn set_fade_grid<G: GraphicsBackend + 'static>(
            ctrl: &mut FrameController<G>,
            app_fade_grid: bool,
        ) {
            ctrl.app_fade_grid = app_fade_grid;
        }

        self.openxr
            .session_data
            .get()
            .comp_data
            .0
            .lock()
            .unwrap()
            .iter_mut()
            .for_each(|ctrl| ctrl.with_any_graphics_mut::<set_fade_grid>(bFadeGridIn));
    }
    fn GetCurrentFadeColor(&self, _bBackground: bool) -> vr::HmdColor_t {
        todo!()
    }
    fn FadeToColor(
        &self,
        _fSeconds: f32,
        _fRed: f32,
        _fGreen: f32,
        _fBlue: f32,
        _fAlpha: f32,
        _bBackground: bool,
    ) {
        crate::warn_unimplemented!("FadeToColor");
    }
    fn GetCumulativeStats(
        &self,
        _pStats: *mut vr::Compositor_CumulativeStats,
        _nStatsSizeInBytes: u32,
    ) {
        todo!()
    }
    fn GetFrameTimeRemaining(&self) -> f32 {
        crate::warn_unimplemented!("GetFrameTimeRemaining");
        0.0
    }
    fn GetFrameTimings(&self, _pTiming: *mut vr::Compositor_FrameTiming, _nFrames: u32) -> u32 {
        todo!()
    }
    fn GetFrameTiming(&self, timing: *mut vr::Compositor_FrameTiming, _frames_ago: u32) -> bool {
        if timing.is_null() || !timing.is_aligned() {
            return false;
        }

        let size = unsafe { (&raw const (*timing).m_nSize).read() } as usize;
        fn ptr_size<T>(_: *const T) -> usize {
            std::mem::size_of::<T>()
        }
        if size
            < offset_of!(vr::Compositor_FrameTiming, m_HmdPose)
                + ptr_size(unsafe { &raw const (*timing).m_HmdPose })
        {
            return false;
        }
        // We're using raw pointers here because the Compositor_FrameTiming struct can be a
        // varaible size, so we don't want to create a reference to a struct with an incorrect
        // (to us) size, because that would be Undefined Behavior.
        macro_rules! set {
            ($member:ident, $value:expr) => {{
                let ptr = &raw mut (*timing).$member;
                ptr.write_unaligned($value)
            }};
        }

        unsafe {
            // TODO: These values are copy/pasted from OpenComposite, determine if real values are
            // necessary/better
            set!(m_nFrameIndex, self.metrics.index.load(Ordering::Relaxed));
            set!(m_nNumFramePresents, 1);
            set!(m_nNumMisPresented, 0);
            set!(m_nReprojectionFlags, 0);
            set!(m_flSystemTimeInSeconds, self.metrics.time.load());
            set!(m_flPreSubmitGpuMs, 8.0);
            set!(m_flPostSubmitGpuMs, 1.0);
            set!(m_flTotalRenderGpuMs, 9.0);

            set!(m_flCompositorRenderGpuMs, 1.5);
            set!(m_flCompositorRenderCpuMs, 3.0);
            set!(m_flCompositorIdleCpuMs, 0.1);

            set!(
                m_flClientFrameIntervalMs,
                1000.0 / self.openxr.get_refresh_rate()
            );
            set!(m_flPresentCallCpuMs, 0.0);
            set!(m_flWaitForPresentCpuMs, 0.0);
            set!(m_flSubmitFrameMs, 0.0);

            set!(m_flWaitGetPosesCalledMs, 0.0);
            set!(m_flNewPosesReadyMs, 0.0);
            set!(m_flNewFrameReadyMs, 0.0); // second call to IVRCompositor::Submit
            set!(m_flCompositorUpdateStartMs, 0.0);
            set!(m_flCompositorUpdateEndMs, 0.0);
            set!(m_flCompositorRenderStartMs, 0.0);
        }

        true
    }
    fn PostPresentHandoff(&self) {
        #[macros::any_graphics(DynFrameController)]
        fn end_frame<G: GraphicsBackend + 'static>(
            ctrl: &mut FrameController<G>,
            session_data: &SessionData,
            system: &System,
            display_time: xr::Time,
            display_period: i64,
            depth_enabled: bool,
            overlays: Option<&OverlayMan>,
        ) where
            for<'b> &'b crate::overlay::AnySwapchainMap:
                TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
            for<'a> &'a openxr_data::GraphicalSession:
                TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
            <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
        {
            ctrl.end_frame(
                session_data,
                system,
                display_time,
                display_period,
                depth_enabled,
                overlays,
            )
        }

        let session_data = self.openxr.session_data.get();
        let mut frame_lock = session_data.comp_data.0.lock().unwrap();
        let Some(ctrl) = frame_lock.as_mut() else {
            debug!("no frame controller - not presenting frame");
            return;
        };

        if *self.frame_state.lock().unwrap() != FrameState::Begun {
            return;
        }

        trace!("presenting frame");
        let system = self.system.force(|i| System::new(self.openxr.clone(), i));
        let display_time = self.openxr.display_time.get();
        let display_period = self.openxr.display_period_nanos.load(Ordering::Relaxed);
        let depth_enabled = self.openxr.enabled_extensions.khr_composition_layer_depth;
        let overlays = self.overlays.get();

        ctrl.with_any_graphics_mut::<end_frame>((
            &session_data,
            &system,
            display_time,
            display_period,
            depth_enabled,
            overlays.as_deref(),
        ));

        self.frame_state
            .lock()
            .unwrap()
            .advance_to(FrameState::Submitted);

        self.metrics.index.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .time
            .store(self.metrics.system_start.elapsed().as_secs_f64());
        #[cfg(feature = "tracing")]
        {
            tracy_client::frame_mark();
        }
    }
    fn ClearLastSubmittedFrame(&self) {
        crate::warn_unimplemented!("ClearLastSubmittedFrame");
    }
    fn SubmitWithArrayIndex(
        &self,
        _eEye: vr::EVREye,
        _pTexture: *const vr::Texture_t,
        _unTextureArrayIndex: u32,
        _pBounds: *const vr::VRTextureBounds_t,
        _nSubmitFlags: vr::EVRSubmitFlags,
    ) -> vr::EVRCompositorError {
        todo!()
    }

    fn GetSubmitTexture(
        &self,
        _: *mut vr::Texture_t,
        __bindgen_vtable: *mut bool,
        _: vr::EVRCompositorTextureUsage,
        _: *const vr::Texture_t,
        _: *const vr::VRTextureBounds_t,
        _: vr::EVRSubmitFlags,
    ) -> vr::EVRCompositorError {
        crate::warn_unimplemented!("GetSubmitTexture");
        vr::EVRCompositorError::IncompatibleVersion
    }

    fn Submit(
        &self,
        eye: vr::EVREye,
        texture: *const vr::Texture_t,
        bounds: *const vr::VRTextureBounds_t,
        submit_flags: vr::EVRSubmitFlags,
    ) -> vr::EVRCompositorError {
        let bounds = unsafe { bounds.as_ref() }
            .copied()
            .unwrap_or(vr::VRTextureBounds_t {
                uMin: 0.0,
                vMin: 0.0,
                uMax: 1.0,
                vMax: 1.0,
            });

        // Superhot passes crazy bounds on startup.
        if !bounds.valid() {
            return vr::EVRCompositorError::InvalidBounds;
        }

        let Some(texture) = (unsafe { texture.as_ref() }) else {
            return vr::EVRCompositorError::InvalidTexture;
        };

        if !self.focused.is_completed() {
            return vr::EVRCompositorError::DoNotHaveFocus;
        }

        let mut session_lock = self.openxr.session_data.get();
        let mut frame_lock = session_lock.comp_data.0.lock().unwrap();

        let ctrl = match frame_lock.as_mut() {
            Some(ctrl) => ctrl,
            None => {
                drop(frame_lock);
                drop(session_lock);

                if let Err(e) = self.initialize_real_session(texture, bounds) {
                    return e;
                }
                info!("Received game texture, restarted session with new data");

                session_lock = self.openxr.session_data.get();
                frame_lock = session_lock.comp_data.0.lock().unwrap();
                frame_lock.as_mut().unwrap()
            }
        };

        #[macros::any_graphics(DynFrameController)]
        fn submit<G: GraphicsBackend + 'static>(
            ctrl: &mut FrameController<G>,
            session_data: &SessionData,
            eye: vr::EVREye,
            texture: &vr::Texture_t,
            bounds: vr::VRTextureBounds_t,
            flags: vr::EVRSubmitFlags,
        ) -> xr::Result<(), vr::EVRCompositorError>
        where
            for<'d> &'d openxr_data::GraphicalSession:
                TryInto<&'d openxr_data::Session<G::Api>, Error: std::fmt::Display>,
            <G::Api as xr::Graphics>::Format: Eq + std::fmt::Debug,
        {
            let real_texture =
                G::get_texture(texture).ok_or(vr::EVRCompositorError::InvalidTexture)?;
            ctrl.submit_impl(
                session_data,
                eye,
                real_texture,
                texture.eColorSpace,
                bounds,
                flags,
            )
        }

        if let Err(e) = ctrl.with_any_graphics_mut::<submit>((
            &session_lock,
            eye,
            texture,
            bounds,
            submit_flags,
        )) {
            return e;
        }
        vr::EVRCompositorError::None
    }

    fn GetLastPoseForTrackedDeviceIndex(
        &self,
        device_index: vr::TrackedDeviceIndex_t,
        output_pose: *mut vr::TrackedDevicePose_t,
        output_game_pose: *mut vr::TrackedDevicePose_t,
    ) -> vr::EVRCompositorError {
        if output_pose.is_null() && output_game_pose.is_null() {
            return vr::EVRCompositorError::RequestFailed;
        }

        let input = self.input.force(|_| Input::new(self.openxr.clone()));

        let Some(pose) = input.get_device_pose(device_index, None) else {
            return vr::EVRCompositorError::RequestFailed;
        };

        if !output_pose.is_null() {
            unsafe { output_pose.write(pose) };
        }

        if !output_game_pose.is_null() {
            unsafe { output_game_pose.write(pose) };
        }

        vr::EVRCompositorError::None
    }
    fn GetLastPoses(
        &self,
        render_pose_array: *mut vr::TrackedDevicePose_t,
        render_pose_count: u32,
        game_pose_array: *mut vr::TrackedDevicePose_t,
        game_pose_count: u32,
    ) -> vr::EVRCompositorError {
        tracy_span!("GetLastPoses impl");
        if render_pose_count == 0 {
            return vr::EVRCompositorError::None;
        }
        let render_poses = unsafe {
            std::slice::from_raw_parts_mut(render_pose_array, render_pose_count as usize)
        };
        self.input
            .force(|_| Input::new(self.openxr.clone()))
            .get_poses(render_poses, None);

        // Not entirely sure how the game poses are supposed to differ from the render poses,
        // but a lot of games use the game pose array for controller positions.
        if game_pose_count > 0 {
            let game_poses = unsafe {
                std::slice::from_raw_parts_mut(game_pose_array, game_pose_count as usize)
            };
            assert!(game_poses.len() <= render_poses.len());
            game_poses.copy_from_slice(&render_poses[0..game_poses.len()]);
        }

        vr::EVRCompositorError::None
    }

    fn WaitGetPoses(
        &self,
        render_pose_array: *mut vr::TrackedDevicePose_t,
        render_pose_count: u32,
        game_pose_array: *mut vr::TrackedDevicePose_t,
        game_pose_count: u32,
    ) -> vr::EVRCompositorError {
        tracy_span!("WaitGetPoses impl");
        // This should be called every frame - we must regularly poll events
        self.openxr.poll_events();
        self.focused.call_once(|| {});
        {
            let session_data = self.openxr.session_data.get();
            let timing_mode = *self.timing_mode.lock().unwrap();
            if matches!(
                timing_mode,
                vr::EVRCompositorTimingMode::Implicit
                    | vr::EVRCompositorTimingMode::Explicit_RuntimePerformsPostPresentHandoff
            ) && *self.frame_state.lock().unwrap() == FrameState::Begun
            {
                self.PostPresentHandoff();
            }

            if *self.frame_state.lock().unwrap() == FrameState::Waited {
                // discard frame
                self.maybe_begin_frame(&session_data);
            }
            self.maybe_wait_frame(&session_data);

            if timing_mode == vr::EVRCompositorTimingMode::Implicit {
                self.maybe_begin_frame(&session_data);
            }
        }
        if let Some(system) = self.system.get() {
            system.reset_views();
        }
        if let Some(input) = self.input.get() {
            input.frame_start_update();
        }

        self.GetLastPoses(
            render_pose_array,
            render_pose_count,
            game_pose_array,
            game_pose_count,
        )
    }

    fn GetTrackingSpace(&self) -> vr::ETrackingUniverseOrigin {
        self.openxr.get_tracking_space()
    }

    fn SetTrackingSpace(&self, origin: vr::ETrackingUniverseOrigin) {
        self.openxr.set_tracking_space(origin);
    }
}

impl vr::IVRCompositor021On022 for Compositor {
    fn SetExplicitTimingMode(&self, explicit: bool) {
        let mode = if explicit {
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff
        } else {
            vr::EVRCompositorTimingMode::Implicit
        };

        <Self as vr::IVRCompositor028_Interface>::SetExplicitTimingMode(self, mode);
    }
}

impl vr::IVRCompositor016On018 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut vr::vr_1_0_3::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v1.0.3)");
        false
    }
}

impl vr::IVRCompositor015On016 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut openvr::vr_1_0_1::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v1.0.1)");
        false
    }
    fn RequestScreenshot(
        &self,
        _screenshot_type: vr::EVRScreenshotType,
        _dest_file_name: *const std::ffi::c_char,
        _vr_dest_file_name: *const std::ffi::c_char,
    ) -> vr::EVRCompositorError {
        todo!("RequestScreenshot (v1.0.1)");
    }
    fn GetCurrentScreenshotType(&self) -> vr::EVRScreenshotType {
        todo!("GetCurrentScreenshotType (v1.0.1)");
    }
}

impl vr::IVRCompositor014On015 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut vr::vr_0_9_20::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v0.9.20)");
        false
    }
}

impl vr::IVRCompositor012On014 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut openvr::vr_0_9_17::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v0.9.17)");
        false
    }
}

impl vr::IVRCompositor011On012 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut openvr::vr_0_9_15::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v0.9.15)");
        false
    }
}

impl vr::IVRCompositor009On011 for Compositor {
    fn GetFrameTiming(
        &self,
        _timing: *mut vr::vr_0_9_12::Compositor_FrameTiming,
        _frames_ago: u32,
    ) -> bool {
        crate::warn_unimplemented!("GetFrameTiming (v0.9.12)");
        false
    }
}

#[derive(Copy, Clone, Default)]
struct SubmittedEye {
    extent: xr::Extent2Di,
    flip_vertically: bool,
}

/// Depth submission outcome: distinguishes a failed runtime `end()`
/// (frame is terminal — re-ending without a new `begin` is invalid and
/// the plain path's `unwrap()` would panic) from ordinary gate misses
/// (caller falls back to color-only `end()`).
#[derive(PartialEq, Eq)]
enum DepthEnd {
    WithDepth,
    Plain,
    /// No regression test: fakexr's end() never fails, so this arm is
    /// reachable only against a real rejecting runtime.
    EndFailed,
}

struct SwapchainData<G: xr::Graphics> {
    swapchain: xr::Swapchain<G>,
    info: xr::SwapchainCreateInfo<G>,
    initial_format: G::Format,
}

/// Same-generation association margin: covers normal present-to-handoff
/// latency plus clock granularity, tuned later from heartbeat age data.
/// Deliberately permissive — false-reject kills the feature silently,
/// while ±1-frame impostors are rejected exactly by the monotonic rule.
const DEPTH_MARGIN_NS: i64 = 2_000_000;
/// Bounded depth-swapchain image acquisition: never stall the app thread
/// on a wedged runtime. Durations are logged (avg/max) on the heartbeat.
const DEPTH_ACQUIRE_TIMEOUT_NS: i64 = 5_000_000;
/// Bounded fence-FD poll for producer-copy completion. EXPERIMENTAL 20ms
/// (up from 5ms): under heavy scene load the copy retires in milliseconds,
/// not microseconds — queue backlog, not bytes — and 5ms starved every
/// frame in-aircraft while the bay flowed at ~3.4ms avg. Re-tune from
/// heartbeat poll[avg/max] data; see the copy-latency discussion.
const DEPTH_FENCE_POLL_TIMEOUT_MS: libc::c_int = 20;
/// EXP12 unsync consumer sample (E12 payoff, torn-tolerant): after the pin,
/// probe the fence with a timeout-0 poll first. If PENDING on a fresh
/// generation, skip the 20ms wait and import_and_copy immediately (GPU-side
/// sample of the GENERAL slot). Hardcoded ON for this experiment; no envs.
const DEPTH_UNSYNC_ON: bool = true;
/// Cooldown between depth swapchain creation retries after a failure.
const DEPTH_SWAPCHAIN_RETRY_COOLDOWN: Duration = Duration::from_secs(2);
/// Heartbeat interval for the depth status line.
const DEPTH_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// Consecutive viable-but-frozen frames before the stalled-producer warn.
/// Frame-counted (not clocked) so hitches and pauses can't trip it.
/// Logging only — never a control input.
const DEPTH_FROZEN_TRIP_FRAMES: u32 = 30;
/// Bound on a held depth pin, in end_frames (~1s at 90Hz). A GL copy fence
/// that never completes past this releases its pin anyway (one torn frame
/// at most): liveness over the pin, consistent with the
/// freshness-over-integrity tradeoff. Logging only beyond the release.
const DEPTH_PIN_MAX_HOLD_FRAMES: u32 = 90;
/// Display period fallback when the runtime hasn't reported one yet.
const DEPTH_FALLBACK_PERIOD_NS: i64 = 11_111_111;

/// Clamp the runtime's display period into a sane window before the age
/// gate. A bogus huge period (pause/transient) would widen the window
/// into accepting arbitrarily stale depth on first arm; a bogus tiny one
/// is meaningless. Legit VR periods sit ~8–17ms (120–60Hz). Pure helper
/// so the clamp is unit-testable.
fn depth_window_period(display_period_ns: i64) -> i64 {
    /// 250Hz — nothing legitimate refreshes faster.
    const MIN_PERIOD_NS: i64 = 4_000_000;
    /// 20Hz — slower means paused or hitching; gate narrow (plain) rather
    /// than accept stale.
    const MAX_PERIOD_NS: i64 = 50_000_000;
    if display_period_ns <= 0 {
        DEPTH_FALLBACK_PERIOD_NS
    } else {
        display_period_ns.clamp(MIN_PERIOD_NS, MAX_PERIOD_NS)
    }
}

struct FrameController<G: GraphicsBackend> {
    stream: xr::FrameStream<G::Api>,
    waiter: xr::FrameWaiter,
    swapchain_data: Option<SwapchainData<G::Api>>,
    image_index: usize,
    image_acquired: bool,
    should_render: bool,
    app_suspend_render: bool,
    app_fade_grid: bool,
    eyes_submitted: [Option<SubmittedEye>; 2],
    submitting_null: bool,
    backend: G,
    depth_swapchain: Option<SwapchainData<G::Api>>,
    depth_feed: Option<DepthFeed>,
    /// Highest generation armed into a depth layer. Monotonic within an
    /// epoch; reset to 0 on epoch change/restart (safe direction: any
    /// publish re-arms).
    last_armed_gen: u64,
    /// SHM slot currently pinned for our in-flight copy, if any. At most
    /// one — the pin is the backpressure, bounding us to one outstanding
    /// copy (no fence vec needed). Unpinned on fence completion (polled at
    /// next end_frame entry), on every plain fallback, and on teardown.
    pinned_slot: Option<usize>,
    /// End_frames since the current pin was taken. Ages out a held pin
    /// that never completes (see `poll_prior_depth_pin`): without this, a
    /// GL fence that neither signals nor fails would pin its slot forever
    /// and the pin_held gate would refuse every later sample — the only
    /// permanent self-sustaining stall in the design. Reset on every
    /// pin take and every release.
    pin_hold_run: u32,
    /// Consecutive viable reads with no generation advance. Trip warn at
    /// [`DEPTH_FROZEN_TRIP_FRAMES`]; reset on advance.
    frozen_run: u32,
    frozen_tripped: bool,
    /// Consecutive viable reads with identical generation while still unarmed
    /// (`last_armed_gen == 0`). The post-arm frozen detector never trips
    /// before the first arm (any `gen > 0` passes the repeat gate), so a
    /// producer frozen from startup would otherwise stay silent behind
    /// window rejects. Same trip threshold; logging only, never control.
    prearm_run: u32,
    prearm_tripped: bool,
    /// Last read outcome, for the dead-to-alive transition log. Starts true
    /// so cold-start absence (normal) stays silent.
    last_read_ok: bool,
    first_arm_logged: bool,
    depth_import_off_logged: bool,
    depth_format_off_logged: bool,
    depth_dims_logged: bool,
    depth_end_fail_warned: bool,
    depth_next_retry: Option<Instant>,
    hb_start: Instant,
    hb_frames: u64,
    hb_depth: u64,
    hb_read_absent: u64,
    hb_read_size: u64,
    hb_read_torn: u64,
    hb_read_magic: u64,
    hb_read_version: u64,
    hb_nonviable: u64,
    hb_time_zero: u64,
    hb_repeat: u64,
    hb_alloc: u64,
    hb_clip: u64,
    hb_window: u64,
    hb_dims: u64,
    hb_flip: u64,
    hb_no_context: u64,
    hb_acquire_fail: u64,
    hb_acquire_timeout: u64,
    hb_copy_fail: u64,
    hb_pin_fail: u64,
    hb_poll_timeout: u64,
    hb_poll_dead: u64,
    hb_unsync: u64,
    hb_pin_held: u64,
    hb_swapchain: u64,
    hb_epoch_raced: u64,
    hb_end_fail: u64,
    evt_timeout_logged: bool,
    evt_dead_logged: bool,
    evt_unsync_logged: bool,
    evt_pinheld_logged: bool,
    evt_copyfail_logged: bool,
    evt_epochrace_logged: bool,
    evt_depth_streak: u32,
    hb_depth_swapchains: u64,
    age_stats: AgeStats,
    acq_count: u64,
    acq_sum_ns: u64,
    acq_max_ns: u64,
    poll_count: u64,
    poll_sum_ns: u64,
    poll_max_ns: u64,
    /// Last observed header identity (updated on every successful SHM read:
    /// the end_frame-entry snapshot before any gate, plus the post-copy
    /// epoch re-read). State, not a counter: never reset. Lets a heartbeat
    /// answer "what dims/epoch/generation are we actually seeing" even when
    /// an earlier gate (e.g. window) rejects every frame before dims match.
    last_seen_gen: u64,
    last_seen_epoch: u32,
    last_seen_w: u32,
    last_seen_h: u32,
}
supported_backends_enum!(enum DynFrameController: FrameController);

impl<G: GraphicsBackend> FrameController<G> {
    fn init_swapchain(
        session_data: &SessionData,
        create_info: &mut xr::SwapchainCreateInfo<G::Api>,
        backend: &mut G,
    ) -> (xr::Swapchain<G::Api>, <G::Api as xr::Graphics>::Format)
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
    {
        assert!(
            is_valid_swapchain_info(create_info),
            "Recreating swapchain with invalid dimensions {}x{}",
            create_info.width,
            create_info.height
        );

        let initial_format = create_info.format;
        session_data.check_format::<G>(create_info);

        let swapchain = session_data
            .create_swapchain(create_info)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to create swapchain: {err} (info: {:#?})",
                    [
                        ("create_flags", format!("{:?}", create_info.create_flags)),
                        ("width", create_info.width.to_string()),
                        ("height", create_info.height.to_string()),
                        ("sample_count", create_info.sample_count.to_string())
                    ]
                )
            });

        let images = swapchain
            .enumerate_images()
            .expect("Failed to enumerate swapchain images");

        backend.store_swapchain_images(images, create_info.format);
        debug!(
            "Created new swapchain: {}x{}, format = {:?}",
            create_info.width, create_info.height, create_info.format
        );

        (swapchain, initial_format)
    }

    fn new(
        session_data: &SessionData,
        waiter: xr::FrameWaiter,
        stream: xr::FrameStream<G::Api>,
        mut backend: G,
        create_info: Option<xr::SwapchainCreateInfo<G::Api>>,
    ) -> Self
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
    {
        let swapchain_data = if let Some(mut info) = create_info {
            is_valid_swapchain_info(&info).then(|| {
                let (swapchain, initial_format) =
                    Self::init_swapchain(session_data, &mut info, &mut backend);
                SwapchainData {
                    swapchain,
                    info,
                    initial_format,
                }
            })
        } else {
            None
        };

        Self {
            stream,
            waiter,
            swapchain_data,
            image_index: 0,
            image_acquired: false,
            should_render: false,
            app_suspend_render: false,
            app_fade_grid: false,
            eyes_submitted: Default::default(),
            submitting_null: false,
            backend,
            depth_swapchain: None,
            depth_feed: None,
            last_armed_gen: 0,
            pinned_slot: None,
            pin_hold_run: 0,
            frozen_run: 0,
            frozen_tripped: false,
            prearm_run: 0,
            prearm_tripped: false,
            last_read_ok: true,
            first_arm_logged: false,
            depth_import_off_logged: false,
            depth_format_off_logged: false,
            depth_dims_logged: false,
            depth_end_fail_warned: false,
            depth_next_retry: None,
            hb_start: Instant::now(),
            hb_frames: 0,
            hb_depth: 0,
            hb_read_absent: 0,
            hb_read_size: 0,
            hb_read_torn: 0,
            hb_read_magic: 0,
            hb_read_version: 0,
            hb_nonviable: 0,
            hb_time_zero: 0,
            hb_repeat: 0,
            hb_alloc: 0,
            hb_clip: 0,
            hb_window: 0,
            hb_dims: 0,
            hb_flip: 0,
            hb_no_context: 0,
            hb_acquire_fail: 0,
            hb_acquire_timeout: 0,
            hb_copy_fail: 0,
            hb_pin_fail: 0,
            hb_poll_timeout: 0,
            hb_poll_dead: 0,
            hb_unsync: 0,
            hb_pin_held: 0,
            hb_swapchain: 0,
            hb_epoch_raced: 0,
            hb_end_fail: 0,
            evt_timeout_logged: false,
            evt_dead_logged: false,
            evt_unsync_logged: false,
            evt_pinheld_logged: false,
            evt_copyfail_logged: false,
            evt_epochrace_logged: false,
            evt_depth_streak: 0,
            hb_depth_swapchains: 0,
            age_stats: AgeStats::new(),
            acq_count: 0,
            acq_sum_ns: 0,
            acq_max_ns: 0,
            poll_count: 0,
            poll_sum_ns: 0,
            poll_max_ns: 0,
            last_seen_gen: 0,
            last_seen_epoch: 0,
            last_seen_w: 0,
            last_seen_h: 0,
        }
    }

    fn recreate_swapchain(
        &mut self,
        session_data: &SessionData,
        mut create_info: xr::SwapchainCreateInfo<G::Api>,
    ) where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
    {
        let (swapchain, initial_format) =
            Self::init_swapchain(session_data, &mut create_info, &mut self.backend);

        self.swapchain_data = Some(SwapchainData {
            swapchain,
            info: create_info,
            initial_format,
        });
        self.acquire_swapchain_image();
        self.eyes_submitted = Default::default();
        // Color geometry is the depth size source; a recreate invalidates
        // any armed depth (dims, imports, pins) — reset to unarmed.
        self.drop_depth_state();
    }

    /// Release a held pin (if any), drop FD imports and the depth
    /// swapchain, and reset arming to unarmed. Safe with no feed/segment:
    /// unpin on a vanished segment is a no-op false. Counters, stats, and
    /// the heartbeat window survive recreates.
    fn drop_depth_state(&mut self) {
        if let (Some(feed), Some(slot)) = (self.depth_feed.as_ref(), self.pinned_slot) {
            feed.unpin_slot(slot);
        }
        self.pinned_slot = None;
        self.pin_hold_run = 0;
        self.backend.drop_depth_imports();
        self.depth_swapchain = None;
        self.last_armed_gen = 0;
        self.frozen_run = 0;
        self.frozen_tripped = false;
        self.prearm_run = 0;
        self.prearm_tripped = false;
    }

    fn acquire_swapchain_image(&mut self) {
        let swapchain = &mut self
            .swapchain_data
            .as_mut()
            .expect("Can't acquire swapchain image with no swapchain!")
            .swapchain;

        self.image_index = swapchain
            .acquire_image()
            .expect("Failed to acquire swapchain image") as usize;

        trace!("waiting image");
        {
            tracy_span!("wait swapchain image");
            swapchain
                .wait_image(xr::Duration::INFINITE)
                .expect("Failed to wait for swapchain image");
        }

        self.image_acquired = true;
    }

    fn wait_frame(&mut self) -> (xr::Time, i64) {
        let frame_state = {
            tracy_span!("wait frame");
            self.waiter.wait().unwrap()
        };
        self.should_render = frame_state.should_render && !self.app_suspend_render;
        (
            frame_state.predicted_display_time,
            frame_state.predicted_display_period.as_nanos(),
        )
    }

    fn begin_frame(&mut self) {
        if self.image_acquired {
            tracy_span!("release old swapchain image");
            self.swapchain_data
                .as_mut()
                .expect("Image is acquired, yet we have no swapchain?")
                .swapchain
                .release_image()
                .unwrap();
        }

        if self.swapchain_data.is_some() {
            self.acquire_swapchain_image();
        }

        {
            tracy_span!("begin frame");
            self.stream.begin().expect("Couldn't begin frame");
        }
        self.eyes_submitted = [None; 2];
        self.submitting_null = false;
        trace!("frame begin");
    }

    fn submit_impl(
        &mut self,
        session_data: &SessionData,
        eye: vr::EVREye,
        texture: G::OpenVrTexture,
        color_space: vr::EColorSpace,
        bounds: vr::VRTextureBounds_t,
        submit_flags: vr::EVRSubmitFlags,
    ) -> Result<(), vr::EVRCompositorError>
    where
        <G::Api as xr::Graphics>::Format: Eq,
        for<'b> &'b openxr_data::GraphicalSession:
            TryInto<&'b openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
    {
        // No Man's Sky does this.
        if self.eyes_submitted[eye as usize].is_some() {
            return Err(vr::EVRCompositorError::AlreadySubmitted);
        }

        self.eyes_submitted[eye as usize] = if self.should_render {
            // Make sure our image dimensions haven't changed.
            let new_info = self
                .backend
                .swapchain_info_for_texture(texture, bounds, color_space);

            is_valid_swapchain_info(&new_info)
                .then(|| {
                    assert!(
                        !self.submitting_null,
                        "App submitted a null texture and a normal texture in the same frame"
                    );

                    if !self.swapchain_data.as_ref().is_some_and(|data| {
                        is_usable_swapchain(&data.info, data.initial_format, &new_info)
                    }) {
                        info!("recreating swapchain (for {eye:?})");
                        self.recreate_swapchain(session_data, new_info);
                    }

                    SubmittedEye {
                        extent: self.backend.copy_texture_to_swapchain(
                            eye,
                            texture,
                            color_space,
                            bounds,
                            self.image_index,
                            submit_flags,
                        ),
                        flip_vertically: bounds.vertically_flipped(),
                    }
                })
                .or_else(|| {
                    trace!("submitting null this frame");
                    self.submitting_null = true;
                    Some(Default::default())
                })
        } else {
            Some(Default::default())
        };

        trace!("submitted {eye:?}");
        if self.eyes_submitted.iter().all(|eye| eye.is_some()) {
            // If the session was restarted while no frame was begun, nothing has been acquired from the new swapchain yet, so there is nothing to release
            if self.image_acquired
                && let Some(data) = self.swapchain_data.as_mut()
            {
                trace!("releasing image");
                data.swapchain.release_image().unwrap();
            }
            self.image_acquired = false;
        }

        Ok(())
    }

    fn end_frame(
        &mut self,
        session_data: &SessionData,
        system: &System,
        display_time: xr::Time,
        display_period: i64,
        depth_enabled: bool,
        overlays: Option<&OverlayMan>,
    ) where
        for<'b> &'b crate::overlay::AnySwapchainMap:
            TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        // Deferred unpin first: a completed prior copy frees its pin even
        // on frames that go plain, so a slow GPU never wedges the feed.
        self.poll_prior_depth_pin();

        let depth_end = self.try_end_frame_with_depth(
            session_data,
            system,
            display_time,
            display_period,
            depth_enabled,
            overlays,
        );
        self.hb_frames = self.hb_frames.saturating_add(1);
        if depth_end == DepthEnd::WithDepth {
            self.hb_depth = self.hb_depth.saturating_add(1);
        }
        self.maybe_log_depth_heartbeat();
        if depth_end != DepthEnd::Plain {
            // WithDepth: submitted. EndFailed: the failed end() already
            // concluded the frame — re-ending without a new begin is
            // invalid (and the plain path unwraps), so stop here.
            return;
        }

        let mut proj_layer_views = Vec::new();

        if self.should_render
            && !self.submitting_null
            && self.eyes_submitted.iter().all(|eye| eye.is_some())
        {
            let swapchain_data = self
                .swapchain_data
                .as_ref()
                .expect("Swapchain data unexpectedly invalid on submit");

            let crate::system::ViewData { flags, views } =
                system.get_views(session_data.current_origin_as_reference_space());
            proj_layer_views = views
                .into_iter()
                .enumerate()
                .map(|(eye_index, view)| {
                    let pose = xr::Posef {
                        orientation: if flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
                            view.pose.orientation
                        } else {
                            xr::Quaternionf::IDENTITY
                        },
                        position: if flags.contains(xr::ViewStateFlags::POSITION_VALID) {
                            view.pose.position
                        } else {
                            xr::Vector3f::default()
                        },
                    };

                    let SubmittedEye {
                        extent,
                        flip_vertically,
                    } = self.eyes_submitted[eye_index]
                        .unwrap_or_else(|| panic!("Eye {eye_index} has not been submitted!"));
                    let mut fov = view.fov;
                    if flip_vertically {
                        std::mem::swap(&mut fov.angle_up, &mut fov.angle_down);
                    }

                    let sub_image = xr::SwapchainSubImage::new()
                        .swapchain(&swapchain_data.swapchain)
                        .image_array_index(eye_index as u32)
                        .image_rect(xr::Rect2Di {
                            extent,
                            offset: xr::Offset2Di::default(),
                        });

                    xr::CompositionLayerProjectionView::new()
                        .fov(fov)
                        .pose(pose)
                        .sub_image(sub_image)
                })
                .collect()
        }

        let mut proj_layer = None;
        if !proj_layer_views.is_empty() {
            trace!("projection layer present");
            proj_layer = Some(
                xr::CompositionLayerProjection::new()
                    .space(session_data.tracking_space())
                    .views(&proj_layer_views),
            );
        }

        let mut layers: Vec<&xr::CompositionLayerBase<_>> = Vec::new();
        if let Some(l) = proj_layer.as_ref() {
            layers.push(l);
        }
        let overlay_layers;
        if let Some(overlay_man) = overlays {
            overlay_layers = overlay_man.get_layers(session_data, self.app_fade_grid);
            layers.extend(overlay_layers.iter().map(Deref::deref));
        }

        self.stream
            .end(display_time, xr::EnvironmentBlendMode::OPAQUE, &layers)
            .unwrap();

        trace!("frame submitted");
    }

    /// Poll the outstanding depth-copy fence (zero timeout, never blocks).
    /// On completion the pin is released; prompt unpin keeps the producer
    /// at full rate. Runs even on plain frames. A hold past
    /// [`DEPTH_PIN_MAX_HOLD_FRAMES`] releases anyway with a warn (a fence
    /// that neither signals nor fails would otherwise stall depth
    /// permanently behind the pin_held gate); the next copy supersedes the
    /// abandoned fence like any skipped generation.
    fn poll_prior_depth_pin(&mut self) {
        let Some(slot) = self.pinned_slot else {
            return;
        };
        if self.backend.poll_depth_copy_complete() {
            if let Some(feed) = self.depth_feed.as_ref() {
                feed.unpin_slot(slot);
            }
            self.pinned_slot = None;
            self.pin_hold_run = 0;
            return;
        }
        self.pin_hold_run = self.pin_hold_run.saturating_add(1);
        if self.pin_hold_run >= DEPTH_PIN_MAX_HOLD_FRAMES {
            warn!(
                "depth copy fence never completed after {} frames; releasing pin on slot {} (one torn frame at most)",
                self.pin_hold_run, slot,
            );
            if let Some(feed) = self.depth_feed.as_ref() {
                feed.unpin_slot(slot);
            }
            self.pinned_slot = None;
            self.pin_hold_run = 0;
        }
    }

    /// Release a held pin immediately (failure paths with no GPU work
    /// outstanding — nothing to wait for, so no reason to hold).
    fn unpin_now(&mut self) {
        if let (Some(feed), Some(slot)) = (self.depth_feed.as_ref(), self.pinned_slot) {
            feed.unpin_slot(slot);
        }
        self.pinned_slot = None;
        self.pin_hold_run = 0;
    }

    fn note_read_skip(&mut self, skip: ReadSkip) {
        let c = match skip {
            ReadSkip::Absent => &mut self.hb_read_absent,
            ReadSkip::WrongSize => &mut self.hb_read_size,
            ReadSkip::Torn => &mut self.hb_read_torn,
            ReadSkip::BadMagic => &mut self.hb_read_magic,
            ReadSkip::BadVersion => &mut self.hb_read_version,
        };
        *c = c.saturating_add(1);
    }

    fn note_read_ok(&mut self, ok: bool) {
        if ok && !self.last_read_ok {
            info!("depth feed alive");
        }
        self.last_read_ok = ok;
    }

    fn sample_acquire_ns(&mut self, ns: u64) {
        self.acq_count = self.acq_count.saturating_add(1);
        self.acq_sum_ns = self.acq_sum_ns.saturating_add(ns);
        self.acq_max_ns = self.acq_max_ns.max(ns);
    }

    fn sample_poll_ns(&mut self, ns: u64) {
        self.poll_count = self.poll_count.saturating_add(1);
        self.poll_sum_ns = self.poll_sum_ns.saturating_add(ns);
        self.poll_max_ns = self.poll_max_ns.max(ns);
    }

    /// Depth path for `end_frame`: single late copy of the newest published
    /// generation, chained as `CompositionLayerDepthInfoKHR` views. No waits
    /// beyond bounded acquire (5ms) + bounded fence poll (20ms, experimental;
    /// see const); the GL copy
    /// itself is queued with its own FenceSync, never a semaphore wait (v4
    /// fence-FD design). Worst case both bounds hit back-to-back (~25ms on
    /// a wedged producer); typical is immediate. Lower the poll bound if
    /// heartbeat `poll[avg]` ever shows pressure.
    fn try_end_frame_with_depth(
        &mut self,
        session_data: &SessionData,
        system: &System,
        display_time: xr::Time,
        display_period: i64,
        depth_enabled: bool,
        overlays: Option<&OverlayMan>,
    ) -> DepthEnd
    where
        for<'b> &'b crate::overlay::AnySwapchainMap:
            TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        // Plain-frame guards first: no clock, no GL, no SHM work outside
        // the projection branch.
        if !(self.should_render
            && !self.submitting_null
            && self.eyes_submitted.iter().all(|eye| eye.is_some()))
        {
            return DepthEnd::Plain;
        }
        if !depth_enabled || !G::supports_depth() {
            return DepthEnd::Plain;
        }
        if !session_data.is_real_session() {
            return DepthEnd::Plain;
        }
        if !depth_master_enabled() {
            return DepthEnd::Plain;
        }
        if self.depth_feed.is_none()
            && let Some(feed) = DepthFeed::open()
        {
            // Heal pins leaked by a crashed predecessor or an unclean
            // restart: a fresh controller holds no pin, so any set word
            // is stale. Releasing backpressure is always fail-safe
            // toward full rate (worst case we unpin a slot nobody held).
            feed.unpin_slot(0);
            feed.unpin_slot(1);
            self.depth_feed = Some(feed);
        }
        // Receipt at end_frame entry: age measures handoff-minus-present
        // (sub-ms normally), maximizing flip discrimination.
        let receipt = now_mono_ns();
        let prev_seen_gen: u64;
        let ctl = match self.depth_feed.as_ref().map(|f| f.read_control()) {
            Some(Ok(c)) => {
                self.note_read_ok(true);
                // Record before any gate: the heartbeat reports what the
                // header actually contains even when a gate rejects it.
                prev_seen_gen = self.last_seen_gen;
                self.last_seen_gen = c.generation;
                self.last_seen_epoch = c.epoch;
                self.last_seen_w = c.w;
                self.last_seen_h = c.h;
                c
            }
            Some(Err(e)) => {
                self.note_read_skip(e);
                self.note_read_ok(false);
                return DepthEnd::Plain;
            }
            None => return DepthEnd::Plain,
        };
        // Cheap gates first: all pure predicates, no GL, no acquires.
        if !ctl.is_viable() {
            self.hb_nonviable = self.hb_nonviable.saturating_add(1);
            return DepthEnd::Plain;
        }
        if ctl.time_ns == 0 {
            self.hb_time_zero = self.hb_time_zero.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Age sample on every viable, timestamped read (hits and
        // window-misses alike): the distribution, not just submissions,
        // tunes the gate. Zero-stamp and non-viable snapshots are excluded
        // so an unhealthy producer can't skew the phase picture.
        self.age_stats.sample(ctl.age_ns(receipt));
        // Pre-arm stall note (logging only): before the first arm the repeat
        // gate can't trip (`gen > 0 == last_armed` always passes), so a
        // producer frozen from startup hides behind window rejects with a
        // wall-clock-growing age. Count identical generations across viable
        // reads here; same threshold as the post-arm detector.
        if self.last_armed_gen == 0 {
            if ctl.generation == prev_seen_gen {
                self.prearm_run = self.prearm_run.saturating_add(1);
                if self.prearm_run == DEPTH_FROZEN_TRIP_FRAMES && !self.prearm_tripped {
                    self.prearm_tripped = true;
                    warn!(
                        "depth producer stalled (pre-arm): generation frozen at {} for {} frames (no depth armed yet)",
                        ctl.generation, self.prearm_run,
                    );
                }
            } else {
                // Advance (or first sight): heal a tripped note.
                if self.prearm_tripped {
                    self.prearm_tripped = false;
                    info!("depth producer publishing (generation {})", ctl.generation);
                }
                self.prearm_run = 0;
            }
        }
        if ctl.generation <= self.last_armed_gen {
            // Same generation re-sighted: present hasn't landed yet (flip
            // or late present) or the producer stalled. Never re-arm —
            // re-presenting older depth under a newer frame is the one
            // hazardous false-accept, so this degrades to plain.
            self.frozen_run = self.frozen_run.saturating_add(1);
            self.hb_repeat = self.hb_repeat.saturating_add(1);
            if self.frozen_run == DEPTH_FROZEN_TRIP_FRAMES && !self.frozen_tripped {
                self.frozen_tripped = true;
                warn!(
                    "depth producer stalled: generation frozen at {} for {} frames",
                    self.last_armed_gen, self.frozen_run,
                );
            }
            return DepthEnd::Plain;
        }
        self.frozen_run = 0;
        if self.frozen_tripped {
            self.frozen_tripped = false;
            info!("depth producer publishing (generation {})", ctl.generation);
        }
        let bpp: u64 = match ctl.format {
            FORMAT_D32F => 4,
            FORMAT_D32S8 => 8,
            _ => {
                // Unreachable via is_viable, but never trust a snapshot
                // twice: re-check rather than mis-size an import.
                self.hb_nonviable = self.hb_nonviable.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        if !ctl.alloc_covers(bpp) {
            self.hb_alloc = self.hb_alloc.saturating_add(1);
            return DepthEnd::Plain;
        }
        let (near_z, far_z) = match validate_near_far(ctl.near_z, ctl.far_z) {
            Some(v) => v,
            None => {
                self.hb_clip = self.hb_clip.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        let period = depth_window_period(display_period);
        if !is_same_generation(ctl.age_ns(receipt), period, DEPTH_MARGIN_NS) {
            self.hb_window = self.hb_window.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Dims are authoritative per read: never stretch, skip the frame.
        let (cw, ch) = match self.swapchain_data.as_ref() {
            Some(c) => (c.info.width, c.info.height),
            None => {
                // No color swapchain (transient recreate): depth can't size.
                // Bucketed as swapchain setup incomplete, like depth creation
                // failures below — every plain path counts exactly one.
                self.hb_swapchain = self.hb_swapchain.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        if cw == 0 || ctl.w != cw * 2 || ctl.h != ch {
            self.hb_dims = self.hb_dims.saturating_add(1);
            if !self.depth_dims_logged {
                self.depth_dims_logged = true;
                // Info (not debug): a dims mismatch is a persistent
                // configuration-level stall, and the heartbeat carries only
                // the counter — the actual values belong in the log.
                info!(
                    "depth dims {}x{} vs color {cw}x{ch} — plain views until match",
                    ctl.w, ctl.h,
                );
            }
            return DepthEnd::Plain;
        }
        self.depth_dims_logged = false;
        // Conservative starter: depth halves are full-eye renders with no
        // flip of their own; flipped color submits would misalign.
        if self
            .eyes_submitted
            .iter()
            .flatten()
            .any(|s| s.flip_vertically)
        {
            self.hb_flip = self.hb_flip.saturating_add(1);
            return DepthEnd::Plain;
        }
        if !self.backend.depth_fd_import_viable() {
            if !self.depth_import_off_logged {
                self.depth_import_off_logged = true;
                info!("depth off: GL FD import not viable (extensions or entry points)");
            }
            // Persistent copy precondition failure: count per frame like any
            // other plain path (the one-time line above names the cause).
            self.evt_depth_streak = 0;
            self.hb_copy_fail = self.hb_copy_fail.saturating_add(1);
            return DepthEnd::Plain;
        }
        // At most one pin: a still-outstanding prior copy means the GPU is
        // behind, so skip (plain) rather than stall it further. Checked
        // before any runtime or GL work below.
        if self.pinned_slot.is_some() {
            if !self.evt_pinheld_logged {
                self.evt_pinheld_logged = true;
                warn!(
                    "depth sample skipped: prior copy still outstanding (first since last with-depth frame)"
                );
            }
            self.evt_depth_streak = 0;
            self.hb_pin_held = self.hb_pin_held.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Context check before acquire: never strand a runtime buffer on a
        // thread that cannot issue GL. Also before swapchain creation so a
        // context-less frame isn't misbucketed as a swapchain failure.
        if !self.backend.depth_context_current() {
            self.hb_no_context = self.hb_no_context.saturating_add(1);
            return DepthEnd::Plain;
        }
        let eye_w = ctl.eye_w();
        let eye_h = ctl.h;
        if !self.ensure_depth_swapchain(session_data, eye_w, eye_h, ctl.format) {
            self.hb_swapchain = self.hb_swapchain.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Acquire (bounded): the only CPU-blocking depth call on this path.
        // Timed for the heartbeat avg/max.
        let t0 = now_mono_ns();
        enum Acquire {
            Done(usize),
            NoImage,
            TimedOut,
        }
        let acquired = (|| {
            let data = self.depth_swapchain.as_mut()?;
            let idx = match data.swapchain.acquire_image().ok() {
                Some(i) => i as usize,
                // Pool exhausted (nothing acquired, nothing to release) —
                // distinct from a wedged wait below.
                None => return Some(Acquire::NoImage),
            };
            if data
                .swapchain
                .wait_image(xr::Duration::from_nanos(DEPTH_ACQUIRE_TIMEOUT_NS))
                .is_err()
            {
                // A timed-out wait must release the image — otherwise each
                // event permanently consumes one swapchain image until the
                // feed latches off instead of skipping a tick.
                let _ = data.swapchain.release_image();
                return Some(Acquire::TimedOut);
            }
            Some(Acquire::Done(idx))
        })();
        let idx: usize = match acquired {
            Some(Acquire::Done(i)) => i,
            Some(Acquire::NoImage) => {
                self.sample_acquire_ns(now_mono_ns().saturating_sub(t0));
                self.hb_acquire_fail = self.hb_acquire_fail.saturating_add(1);
                return DepthEnd::Plain;
            }
            Some(Acquire::TimedOut) => {
                self.sample_acquire_ns(now_mono_ns().saturating_sub(t0));
                self.hb_acquire_timeout = self.hb_acquire_timeout.saturating_add(1);
                return DepthEnd::Plain;
            }
            None => {
                self.sample_acquire_ns(now_mono_ns().saturating_sub(t0));
                self.hb_acquire_fail = self.hb_acquire_fail.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        self.sample_acquire_ns(now_mono_ns().saturating_sub(t0));
        // Pin FIRST (contract order: pin → poll → sample), from the
        // generation just read. A vanished segment mid-frame fails here.
        // The pin lands late (after gates + bounded acquire), not at the
        // read: holding it across the acquire would stall the producer for
        // no benefit. A same-epoch advance inside that window only costs
        // freshness (the snapshot stays valid and correctly labeled; the
        // window/monotonic gates still apply to it) — never safety.
        // The borrow ends immediately: later steps re-borrow, so `&mut`
        // work (stats, backend, unpin) never fights it.
        let slot = ctl.slot();
        debug_assert!(self.pinned_slot.is_none());
        if !self
            .depth_feed
            .as_ref()
            .is_some_and(|feed| feed.pin_slot(slot))
        {
            Self::release_depth_image(&mut self.depth_swapchain);
            self.hb_pin_fail = self.hb_pin_fail.saturating_add(1);
            return DepthEnd::Plain;
        }
        self.pinned_slot = Some(slot);
        self.pin_hold_run = 0;
        // Structural bound check: at most one pin, ever. Ours is now set;
        // if the OTHER word is also set (our own bug — the startup heal
        // clears predecessors, and single-process flow cannot set both),
        // refuse to hold both: a second tracked pin would overwrite
        // pinned_slot and strand the first permanently, besides stalling
        // the producer. Loud skip instead of silent leak.
        if self
            .depth_feed
            .as_ref()
            .is_some_and(|feed| feed.load_pin(1 - slot) == Some(1))
        {
            warn!(
                "depth both SHM slots pinned (ours {slot}); releasing ours instead of holding both"
            );
            Self::release_depth_image(&mut self.depth_swapchain);
            self.unpin_now();
            self.evt_depth_streak = 0;
            self.hb_pin_held = self.hb_pin_held.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Poll the slot's fence sync_file for producer-copy completion
        // (bounded; the pin keeps the polled number stable — the producer
        // closes and replaces fence FDs only on unpinned reuse, and a
        // vanished segment reports Dead). No semaphore objects exist on
        // either side: the FD snapshots exactly this publish, so skipped
        // generations leave no sync debt behind.
        // EXP12 unsync fast path: timeout-0 probe first. READY proceeds as
        // today; PENDING on a fresh generation skips the 20ms wait and
        // samples immediately (torn-tolerant GPU-side sample). Repeat gens
        // never unsync (repeat gate above guarantees ctl.gen > last_armed
        // here, re-checked defensively). Dims/epoch viability already gated
        // above; the post-copy epoch re-read stays the staleness net.
        let t1 = now_mono_ns();
        let fence_fd = ctl.fence_fds[slot];
        let mut unsync_sampled = false;
        match poll_fence_fd(fence_fd, 0) {
            FencePoll::Ready => {}
            FencePoll::Timeout => {
                let fresh = ctl.generation > self.last_armed_gen;
                if DEPTH_UNSYNC_ON && fresh {
                    unsync_sampled = true;
                    self.hb_unsync = self.hb_unsync.saturating_add(1);
                    if !self.evt_unsync_logged {
                        self.evt_unsync_logged = true;
                        info!(
                            "[xrizer] exp12 unsync: sampling PENDING gen {}",
                            ctl.generation
                        );
                    }
                }
            }
            FencePoll::Dead => {
                self.sample_poll_ns(now_mono_ns().saturating_sub(t1));
                self.unpin_now();
                Self::release_depth_image(&mut self.depth_swapchain);
                if !self.evt_dead_logged {
                    self.evt_dead_logged = true;
                    warn!("depth fence FD dead (first since last with-depth frame)");
                }
                self.evt_depth_streak = 0;
                self.hb_poll_dead = self.hb_poll_dead.saturating_add(1);
                return DepthEnd::Plain;
            }
        }
        if !unsync_sampled {
            match poll_fence_fd(fence_fd, DEPTH_FENCE_POLL_TIMEOUT_MS) {
                FencePoll::Ready => {}
                FencePoll::Timeout => {
                    self.sample_poll_ns(now_mono_ns().saturating_sub(t1));
                    self.unpin_now();
                    Self::release_depth_image(&mut self.depth_swapchain);
                    if !self.evt_timeout_logged {
                        self.evt_timeout_logged = true;
                        warn!(
                            "depth fence poll timed out after {}ms (first since last with-depth frame)",
                            DEPTH_FENCE_POLL_TIMEOUT_MS
                        );
                    }
                    self.evt_depth_streak = 0;
                    self.hb_poll_timeout = self.hb_poll_timeout.saturating_add(1);
                    return DepthEnd::Plain;
                }
                FencePoll::Dead => {
                    self.sample_poll_ns(now_mono_ns().saturating_sub(t1));
                    self.unpin_now();
                    Self::release_depth_image(&mut self.depth_swapchain);
                    if !self.evt_dead_logged {
                        self.evt_dead_logged = true;
                        warn!("depth fence FD dead (first since last with-depth frame)");
                    }
                    self.evt_depth_streak = 0;
                    self.hb_poll_dead = self.hb_poll_dead.saturating_add(1);
                    return DepthEnd::Plain;
                }
            }
        }
        self.sample_poll_ns(now_mono_ns().saturating_sub(t1));
        // Import note: the backend reimports memory when this snapshot's
        // key (epoch, dims, format) differs from its cache. No fresh read
        // sits between the pin and the import — none is needed: we sample
        // only the generation just read (single-snapshot discipline), and
        // the post-copy epoch re-read below is the staleness net that
        // discards a copy made from recycled FDs.
        let res = self.backend.import_and_copy_depth_fd(&DepthFdRequest {
            epoch: ctl.epoch,
            w: ctl.w,
            h: ctl.h,
            vk_format: ctl.format,
            mem_fds: ctl.mem_fds,
            generation: ctl.generation,
            depth_image_index: idx,
            eye_w,
            eye_h,
            alloc_bytes: ctl.alloc_bytes,
        });
        // Release before end (color path releases before end too). Scope
        // discipline: every post-acquire path releases exactly once.
        Self::release_depth_image(&mut self.depth_swapchain);
        if !res.copied {
            // No GPU work outstanding: drop the pin immediately rather than
            // holding a slot for nothing.
            self.unpin_now();
            if !self.evt_copyfail_logged {
                self.evt_copyfail_logged = true;
                warn!("depth GL copy failed (first since last with-depth frame)");
            }
            self.evt_depth_streak = 0;
            self.hb_copy_fail = self.hb_copy_fail.saturating_add(1);
            return DepthEnd::Plain;
        }
        // Post-copy epoch re-read: a recreate raced the copy (stale FDs
        // imported, wrong pixels). Discard rather than mislabel. A
        // successful re-read refreshes the header identity (the heartbeat's
        // seen[] tracks every Ok read, including this staleness net).
        match self.depth_feed.as_ref().map(|f| f.read_control()) {
            Some(Ok(fresh)) => {
                self.last_seen_gen = fresh.generation;
                self.last_seen_epoch = fresh.epoch;
                self.last_seen_w = fresh.w;
                self.last_seen_h = fresh.h;
                if fresh.epoch != ctl.epoch {
                    self.unpin_now();
                    if !self.evt_epochrace_logged {
                        self.evt_epochrace_logged = true;
                        warn!(
                            "depth epoch raced the copy; discarded (first since last with-depth frame)"
                        );
                    }
                    self.evt_depth_streak = 0;
                    self.hb_epoch_raced = self.hb_epoch_raced.saturating_add(1);
                    return DepthEnd::Plain;
                }
            }
            _ => {
                self.unpin_now();
                self.evt_depth_streak = 0;
                self.hb_epoch_raced = self.hb_epoch_raced.saturating_add(1);
                return DepthEnd::Plain;
            }
        }
        // Armed by this call: both halves were copied atomically, so this
        // chains both eyes or neither and performs no further GL work.
        let (min_depth, max_depth) = (MIN_DEPTH, MAX_DEPTH);
        let depth_data = match self.depth_swapchain.as_ref() {
            Some(d) => d,
            None => {
                // Defensive: verified Some at acquire; single-threaded flow
                // can't clear it. Bucketed so every plain path counts one.
                self.unpin_now();
                self.evt_depth_streak = 0;
                self.hb_copy_fail = self.hb_copy_fail.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        let color_data = match self.swapchain_data.as_ref() {
            Some(c) => c,
            None => {
                // Defensive: verified Some at the dims gate. Same bucket.
                self.unpin_now();
                self.evt_depth_streak = 0;
                self.hb_copy_fail = self.hb_copy_fail.saturating_add(1);
                return DepthEnd::Plain;
            }
        };
        let crate::system::ViewData { flags, views } =
            system.get_views(session_data.current_origin_as_reference_space());
        let eyes = self.eyes_submitted;

        let mut color_subs = Vec::with_capacity(2);
        let mut depth_subs = Vec::with_capacity(2);
        let mut poses = Vec::with_capacity(2);
        let mut fovs = Vec::with_capacity(2);
        for (eye_index, view) in views.into_iter().enumerate() {
            let pose = xr::Posef {
                orientation: if flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
                    view.pose.orientation
                } else {
                    xr::Quaternionf::IDENTITY
                },
                position: if flags.contains(xr::ViewStateFlags::POSITION_VALID) {
                    view.pose.position
                } else {
                    xr::Vector3f::default()
                },
            };
            let SubmittedEye {
                extent,
                flip_vertically: _,
            } = eyes[eye_index]
                .unwrap_or_else(|| panic!("Eye {eye_index} has not been submitted!"));
            // Flip was gated above (conservative starter): no FOV swap here.
            poses.push(pose);
            fovs.push(view.fov);
            color_subs.push(
                xr::SwapchainSubImage::new()
                    .swapchain(&color_data.swapchain)
                    .image_array_index(eye_index as u32)
                    .image_rect(xr::Rect2Di {
                        extent,
                        offset: xr::Offset2Di::default(),
                    }),
            );
            depth_subs.push(
                xr::SwapchainSubImage::new()
                    .swapchain(&depth_data.swapchain)
                    .image_array_index(eye_index as u32)
                    .image_rect(xr::Rect2Di {
                        extent: xr::Extent2Di {
                            width: eye_w as i32,
                            height: eye_h as i32,
                        },
                        offset: xr::Offset2Di::default(),
                    }),
            );
        }
        if color_subs.len() != 2 || depth_subs.len() != 2 {
            // Defensive: the depth layer hardcodes two eyes. A runtime
            // reporting any other view count can't chain depth; plain.
            self.unpin_now();
            self.evt_depth_streak = 0;
            self.hb_copy_fail = self.hb_copy_fail.saturating_add(1);
            return DepthEnd::Plain;
        }

        let depth_infos: Vec<xr::sys::CompositionLayerDepthInfoKHR> = (0..2)
            .map(|i| xr::sys::CompositionLayerDepthInfoKHR {
                ty: xr::sys::CompositionLayerDepthInfoKHR::TYPE,
                next: std::ptr::null(),
                sub_image: *depth_subs[i].as_raw(),
                min_depth,
                max_depth,
                near_z,
                far_z,
            })
            .collect();
        // near/far validated at gate time (near < far guaranteed).
        let sys_views: Vec<xr::sys::CompositionLayerProjectionView> = (0..2)
            .map(|i| xr::sys::CompositionLayerProjectionView {
                ty: xr::sys::CompositionLayerProjectionView::TYPE,
                next: &depth_infos[i] as *const _ as *const std::ffi::c_void,
                pose: poses[i],
                fov: fovs[i],
                sub_image: *color_subs[i].as_raw(),
            })
            .collect();
        // SAFETY: from_raw only wraps the structs; lifetimes are preserved
        // because swapchains, views, and depth_infos all outlive end().
        let proj_views: Vec<xr::CompositionLayerProjectionView<G::Api>> = sys_views
            .into_iter()
            .map(|s| unsafe { xr::CompositionLayerProjectionView::from_raw(s) })
            .collect();

        trace!("projection layer present (with depth)");
        let proj_layer = xr::CompositionLayerProjection::new()
            .space(session_data.tracking_space())
            .views(&proj_views);
        let mut layers: Vec<&xr::CompositionLayerBase<_>> = Vec::new();
        layers.push(&proj_layer);
        let overlay_layers;
        if let Some(overlay_man) = overlays {
            overlay_layers = overlay_man.get_layers(session_data, self.app_fade_grid);
            layers.extend(overlay_layers.iter().map(Deref::deref));
        }
        match self
            .stream
            .end(display_time, xr::EnvironmentBlendMode::OPAQUE, &layers)
        {
            Ok(()) => {
                trace!("frame submitted (with depth)");
                self.last_armed_gen = ctl.generation;
                self.prearm_run = 0;
                self.prearm_tripped = false;
                self.depth_end_fail_warned = false;
                self.evt_depth_streak = self.evt_depth_streak.saturating_add(1);
                if self.evt_depth_streak >= 5 {
                    self.evt_timeout_logged = false;
                    self.evt_dead_logged = false;
                    self.evt_pinheld_logged = false;
                    self.evt_copyfail_logged = false;
                    self.evt_epochrace_logged = false;
                }
                if !self.first_arm_logged {
                    self.first_arm_logged = true;
                    info!(
                        "depth armed: {eye_w}x{eye_h}x2 fmt {} near {near_z} far {far_z}",
                        ctl.format,
                    );
                }
                DepthEnd::WithDepth
            }
            Err(e) => {
                // The copy completed (fence still governs the pin), but the
                // runtime rejected the frame: terminal, do NOT re-end (the
                // failed end() concluded the frame; the plain path below
                // must not run). Throttled: a persistently rejecting
                // runtime must not warn per frame. last_armed_gen still
                // advances — the pixels are superseded by newest-wins next
                // frame either way.
                self.last_armed_gen = ctl.generation;
                self.prearm_run = 0;
                self.prearm_tripped = false;
                self.hb_end_fail = self.hb_end_fail.saturating_add(1);
                if !self.depth_end_fail_warned {
                    self.depth_end_fail_warned = true;
                    warn!("depth frame end failed ({e}); frame dropped");
                }
                DepthEnd::EndFailed
            }
        }
    }

    /// Release the depth swapchain image acquired for this frame.
    /// Best-effort: failures are debug-logged (the buffer heals via acquire
    /// cycling, and every path after this one ends the frame either way).
    /// OpenXR releases the oldest acquired image; single-acquire discipline
    /// (exactly one acquire per end_frame, unconditional release below)
    /// makes that ours.
    fn release_depth_image(depth_swapchain: &mut Option<SwapchainData<G::Api>>) {
        if let Some(data) = depth_swapchain.as_mut()
            && data.swapchain.release_image().is_err()
        {
            debug!("depth release_image failed");
        }
    }

    /// Ensure a per-eye depth swapchain for the mirror dims. Recreates on
    /// dims/format change. Never panics (unlike color init): any failure
    /// degrades to plain frames with a 2s retry cooldown.
    fn ensure_depth_swapchain(
        &mut self,
        session_data: &SessionData,
        eye_w: u32,
        eye_h: u32,
        vk_format: i32,
    ) -> bool
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        let Some(want_fmt) = G::depth_format_for_header(vk_format) else {
            return false;
        };
        if let Some(d) = self.depth_swapchain.as_ref()
            && d.info.width == eye_w
            && d.info.height == eye_h
            && d.info.format == want_fmt
        {
            return true;
        }
        if let Some(t) = self.depth_next_retry
            && Instant::now() < t
        {
            return false;
        }
        // No fallback: a remapped color format as depth would be garbage.
        if !session_data.has_swapchain_format::<G::Api>(want_fmt) {
            if !self.depth_format_off_logged {
                self.depth_format_off_logged = true;
                info!("depth off: preferred depth format not advertised; color-only");
            }
            return false;
        }
        let color_info = match self.swapchain_data.as_ref() {
            Some(c) => &c.info,
            None => return false,
        };
        let mut info = match self
            .backend
            .depth_swapchain_create_info(color_info, want_fmt)
        {
            Some(i) => i,
            None => return false,
        };
        info.width = eye_w;
        info.height = eye_h;
        let swapchain = match session_data.create_swapchain(&info) {
            Ok(s) => s,
            Err(e) => {
                warn!("depth swapchain creation failed ({e}); color-only");
                self.depth_next_retry = Some(Instant::now() + DEPTH_SWAPCHAIN_RETRY_COOLDOWN);
                return false;
            }
        };
        let images = match swapchain.enumerate_images() {
            Ok(v) => v,
            Err(e) => {
                warn!("failed to enumerate depth swapchain images ({e})");
                self.depth_next_retry = Some(Instant::now() + DEPTH_SWAPCHAIN_RETRY_COOLDOWN);
                return false;
            }
        };
        self.backend
            .store_depth_swapchain_images(images, info.format);
        info!(
            "Created depth swapchain: {}x{}x2, format = {:?}",
            eye_w, eye_h, info.format,
        );
        self.depth_swapchain = Some(SwapchainData {
            swapchain,
            info,
            initial_format: want_fmt,
        });
        self.depth_next_retry = None;
        self.hb_depth_swapchains = self.hb_depth_swapchains.saturating_add(1);
        true
    }

    /// Throttled heartbeat: one status line per window with the depth/plain
    /// split, every skip bucket, signed age stats, acquire/poll timing, and
    /// the last observed header identity (gen/epoch/dims, never reset). Hot
    /// path pays counter increments only; formatting happens here.
    fn maybe_log_depth_heartbeat(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.hb_start) < DEPTH_HEARTBEAT_INTERVAL {
            return;
        }
        let secs = now.duration_since(self.hb_start).as_secs_f64().max(0.001);
        let plain = self.hb_frames.saturating_sub(self.hb_depth);
        let mean_age = self
            .age_stats
            .mean_ns()
            .map(|m| format!("{:+.1}ms", m / 1_000_000.0))
            .unwrap_or_else(|| "—".to_string());
        let max_age = self
            .age_stats
            .max_ns()
            .map(|m| format!("{:+.1}ms", m as f64 / 1_000_000.0))
            .unwrap_or_else(|| "—".to_string());
        let min_age = self
            .age_stats
            .min_ns()
            .map(|m| format!("{:+.1}ms", m as f64 / 1_000_000.0))
            .unwrap_or_else(|| "—".to_string());
        let acq_avg = if self.acq_count == 0 {
            "—".to_string()
        } else {
            format!(
                "{:.3}ms",
                self.acq_sum_ns as f64 / self.acq_count as f64 / 1_000_000.0
            )
        };
        let poll_avg = if self.poll_count == 0 {
            "—".to_string()
        } else {
            format!(
                "{:.3}ms",
                self.poll_sum_ns as f64 / self.poll_count as f64 / 1_000_000.0
            )
        };
        info!(
            "depth {:.1}/s ({} frames, {} with depth, {} plain) \
             read[absent {} size {} torn {} magic {} version {}] \
             gate[nonviable {} t0 {} repeat {} alloc {} clip {} window {} dims {} flip {}] \
             copy[noctx {} acq_fail {} acq_timeout {} poll_timeout {} poll_dead {} unsync {} copyfail {} pinfail {} pinheld {} swapchain {} epochrace {} endfail {}] \
             swapchains {} age[n {} avg {} max {} min {} neg {:.0}%] acq[avg {} max {:.3}ms] poll[avg {} max {:.3}ms] seen[gen {} epoch {} {}x{}]",
            self.hb_frames as f64 / secs,
            self.hb_frames,
            self.hb_depth,
            plain,
            self.hb_read_absent,
            self.hb_read_size,
            self.hb_read_torn,
            self.hb_read_magic,
            self.hb_read_version,
            self.hb_nonviable,
            self.hb_time_zero,
            self.hb_repeat,
            self.hb_alloc,
            self.hb_clip,
            self.hb_window,
            self.hb_dims,
            self.hb_flip,
            self.hb_no_context,
            self.hb_acquire_fail,
            self.hb_acquire_timeout,
            self.hb_poll_timeout,
            self.hb_poll_dead,
            self.hb_unsync,
            self.hb_copy_fail,
            self.hb_pin_fail,
            self.hb_pin_held,
            self.hb_swapchain,
            self.hb_epoch_raced,
            self.hb_end_fail,
            self.hb_depth_swapchains,
            self.age_stats.count(),
            mean_age,
            max_age,
            min_age,
            self.age_stats.neg_rate() * 100.0,
            acq_avg,
            self.acq_max_ns as f64 / 1_000_000.0,
            poll_avg,
            self.poll_max_ns as f64 / 1_000_000.0,
            self.last_seen_gen,
            self.last_seen_epoch,
            self.last_seen_w,
            self.last_seen_h,
        );
        self.hb_start = now;
        self.hb_frames = 0;
        self.hb_depth = 0;
        self.hb_read_absent = 0;
        self.hb_read_size = 0;
        self.hb_read_torn = 0;
        self.hb_read_magic = 0;
        self.hb_read_version = 0;
        self.hb_nonviable = 0;
        self.hb_time_zero = 0;
        self.hb_repeat = 0;
        self.hb_alloc = 0;
        self.hb_clip = 0;
        self.hb_window = 0;
        self.hb_dims = 0;
        self.hb_flip = 0;
        self.hb_no_context = 0;
        self.hb_acquire_fail = 0;
        self.hb_acquire_timeout = 0;
        self.hb_poll_timeout = 0;
        self.hb_poll_dead = 0;
        self.hb_unsync = 0;
        self.hb_copy_fail = 0;
        self.hb_pin_fail = 0;
        self.hb_pin_held = 0;
        self.hb_swapchain = 0;
        self.hb_epoch_raced = 0;
        self.hb_end_fail = 0;
        self.hb_depth_swapchains = 0;
        self.age_stats.clear();
        self.acq_count = 0;
        self.acq_sum_ns = 0;
        self.acq_max_ns = 0;
        self.poll_count = 0;
        self.poll_sum_ns = 0;
        self.poll_max_ns = 0;
    }
}

pub fn is_usable_swapchain<G: xr::Graphics>(
    current: &xr::SwapchainCreateInfo<G>,
    creation_format: G::Format,
    new: &xr::SwapchainCreateInfo<G>,
) -> bool
where
    G::Format: Eq,
{
    creation_format == new.format
        && current.width >= new.width
        && current.height >= new.height
        && current.array_size == new.array_size
        && current.sample_count == new.sample_count
}

fn is_valid_swapchain_info<G: xr::Graphics>(info: &xr::SwapchainCreateInfo<G>) -> bool {
    info.width > 0 && info.height > 0
}

#[cfg(test)]
pub use tests::FakeGraphicsData;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics_backends::{GraphicsBackend, VulkanData};
    use std::cell::Cell;
    use std::ffi::CStr;
    use std::mem::MaybeUninit;
    use std::thread_local;
    use vr::EVRCompositorError::*;
    use vr::IVRCompositor028_Interface;

    pub struct FakeGraphicsData {
        vk: Arc<VulkanData>,
        swapchain_format: Option<u32>,
    }
    thread_local! {
        static SWAPCHAIN_WIDTH: Cell<u32> = const { Cell::new(10) };
        static SWAPCHAIN_HEIGHT: Cell<u32> = const { Cell::new(10) };
        static SWAPCHAIN_FORMAT: Cell<u32> = const { Cell::new(0) };
    }

    pub enum FakeApi {}
    impl xr::Graphics for FakeApi {
        type Requirements = xr::vulkan::Requirements;
        type SessionCreateInfo = xr::vulkan::SessionCreateInfo;
        type Format = <xr::Vulkan as xr::Graphics>::Format;
        type SwapchainImage = <xr::Vulkan as xr::Graphics>::SwapchainImage;

        fn raise_format(x: i64) -> Self::Format {
            xr::Vulkan::raise_format(x)
        }

        fn lower_format(x: Self::Format) -> i64 {
            xr::Vulkan::lower_format(x)
        }

        fn requirements(
            instance: &openxr::Instance,
            system: openxr::SystemId,
        ) -> openxr::Result<Self::Requirements> {
            xr::Vulkan::requirements(instance, system)
        }

        unsafe fn create_session(
            instance: &openxr::Instance,
            system: openxr::SystemId,
            info: &Self::SessionCreateInfo,
        ) -> openxr::Result<openxr::sys::Session> {
            unsafe { xr::Vulkan::create_session(instance, system, info) }
        }

        fn enumerate_swapchain_images(
            _: &openxr::Swapchain<Self>,
        ) -> openxr::Result<Vec<Self::SwapchainImage>> {
            Ok(Vec::new())
        }
    }

    impl GraphicsBackend for FakeGraphicsData {
        type Api = FakeApi;
        type OpenVrTexture = <VulkanData as GraphicsBackend>::OpenVrTexture;
        type NiceFormat = <VulkanData as GraphicsBackend>::NiceFormat;

        fn to_nice_format(format: <Self::Api as openxr::Graphics>::Format) -> Self::NiceFormat {
            VulkanData::to_nice_format(format)
        }
        fn session_create_info(&self) -> <Self::Api as openxr::Graphics>::SessionCreateInfo {
            self.vk.session_create_info()
        }

        fn get_texture(texture: &openvr::Texture_t) -> Option<Self::OpenVrTexture> {
            VulkanData::get_texture(texture)
        }

        fn swapchain_info_for_texture(
            &self,
            _: Self::OpenVrTexture,
            _: openvr::VRTextureBounds_t,
            _: openvr::EColorSpace,
        ) -> openxr::SwapchainCreateInfo<Self::Api> {
            xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::EMPTY,
                format: SWAPCHAIN_FORMAT.get(),
                sample_count: 1,
                width: SWAPCHAIN_WIDTH.get(),
                height: SWAPCHAIN_HEIGHT.get(),
                face_count: 1,
                array_size: 2,
                mip_count: 1,
            }
        }

        fn store_swapchain_images(
            &mut self,
            _images: Vec<<Self::Api as openxr::Graphics>::SwapchainImage>,
            format: u32,
        ) {
            self.swapchain_format.replace(format);
        }

        fn copy_texture_to_swapchain(
            &self,
            _eye: openvr::EVREye,
            _texture: Self::OpenVrTexture,
            _color_space: vr::EColorSpace,
            _bounds: openvr::VRTextureBounds_t,
            _image_index: usize,
            _submit_flags: openvr::EVRSubmitFlags,
        ) -> openxr::Extent2Di {
            xr::Extent2Di::default()
        }

        fn copy_overlay_to_swapchain(
            &mut self,
            _texture: Self::OpenVrTexture,
            _bounds: openvr::VRTextureBounds_t,
            _image_index: usize,
        ) -> openxr::Extent2Di {
            xr::Extent2Di::default()
        }
    }

    impl FakeGraphicsData {
        fn texture(data: &Arc<VulkanData>) -> vr::Texture_t {
            vr::Texture_t {
                eType: vr::ETextureType::Reserved,
                handle: Arc::into_raw(data.clone()) as _,
                eColorSpace: vr::EColorSpace::Auto,
            }
        }

        pub fn new(texture: &vr::Texture_t) -> Self {
            assert_eq!(texture.eType, vr::ETextureType::Reserved);
            let ptr = texture.handle as *const VulkanData;
            let vk = unsafe {
                Arc::increment_strong_count(ptr);
                Arc::from_raw(ptr)
            };
            Self {
                vk,
                swapchain_format: Option::None,
            }
        }
    }

    struct Fixture {
        comp: Arc<Compositor>,
        vk: Arc<VulkanData>,
    }

    impl Fixture {
        fn new() -> Self {
            let xr = Arc::new(OpenXrData::new(&Injector::default()).unwrap());
            let vk = Arc::new(VulkanData::new_temporary(&xr.instance, xr.system_id));
            let comp = Arc::new(Compositor::new(xr.clone(), &Injector::default()));
            xr.compositor.set(Arc::downgrade(&comp));
            crate::init_logging();

            Self { comp, vk }
        }

        fn wait_get_poses(&self) -> vr::EVRCompositorError {
            self.comp
                .WaitGetPoses(std::ptr::null_mut(), 0, std::ptr::null_mut(), 0)
        }

        fn submit(&self, eye: vr::EVREye) -> vr::EVRCompositorError {
            self.comp.Submit(
                eye,
                &FakeGraphicsData::texture(&self.vk),
                std::ptr::null(),
                vr::EVRSubmitFlags::Default,
            )
        }

        fn ensure_real_session(&self, explicit: bool) {
            // synchronize session
            assert_eq!(self.wait_get_poses(), None);
            if explicit {
                assert_eq!(self.comp.SubmitExplicitTimingData(), None);
            }
            assert_eq!(self.submit(vr::EVREye::Left), None);
            assert_eq!(self.submit(vr::EVREye::Right), None);
            if explicit {
                self.comp.PostPresentHandoff();
            } else {
                assert_eq!(self.wait_get_poses(), None);
            }

            let data = self.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(!ctrl.should_render);
        }

        #[track_caller]
        fn check_frame_state(&self, state: fakexr::FrameState) {
            let session = self.comp.openxr.session_data.get().session.as_raw();
            assert_eq!(fakexr::session_frame_state(session), state);
        }
    }

    #[test]
    fn bad_bounds() {
        let f = Fixture::new();

        let bad_bounds = [
            // bound greater than 1 (Superhot does this)
            vr::VRTextureBounds_t {
                vMin: 0.0,
                vMax: 1.1,
                uMin: 0.0,
                uMax: 1.0,
            },
            // bound greater than 1
            vr::VRTextureBounds_t {
                vMin: 0.0,
                vMax: 1.0,
                uMin: 0.0,
                uMax: 1.1,
            },
            // negative bound
            vr::VRTextureBounds_t {
                vMin: -0.1,
                vMax: 1.0,
                uMin: 0.0,
                uMax: 1.0,
            },
            // min/max equal
            vr::VRTextureBounds_t {
                vMin: 1.0,
                vMax: 1.0,
                uMin: 0.0,
                uMax: 1.0,
            },
        ];

        for bound in bad_bounds {
            assert_eq!(
                f.comp.Submit(
                    vr::EVREye::Left,
                    &FakeGraphicsData::texture(&f.vk),
                    &bound,
                    vr::EVRSubmitFlags::Default,
                ),
                InvalidBounds,
                "Bound didn't return InvalidBounds: {bound:?}"
            );
        }
    }

    #[test]
    fn allow_flipped_bounds() {
        let Fixture { comp, .. } = Fixture::new();

        assert_ne!(
            comp.Submit(
                vr::EVREye::Left,
                std::ptr::null(),
                &vr::VRTextureBounds_t {
                    uMin: 0.0,
                    vMin: 1.0,
                    uMax: 1.0,
                    vMax: 0.0
                },
                vr::EVRSubmitFlags::Default
            ),
            InvalidBounds
        );
    }

    #[test]
    fn error_on_multiple_same_eye_submit() {
        let f = Fixture::new();

        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        assert_eq!(f.submit(vr::EVREye::Left), AlreadySubmitted);

        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
    }

    #[test]
    fn allow_waitgetposes_without_submit() {
        let f = Fixture::new();

        // Enable frame controller
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);

        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.wait_get_poses(), None);
    }

    #[test]
    fn recreate_swapchain() {
        let f = Fixture::new();
        f.ensure_real_session(false);

        let get_swapchain_width = || {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            ctrl.swapchain_data
                .as_ref()
                .expect("swapchain info missing")
                .info
                .width
        };

        let old_width = get_swapchain_width();
        SWAPCHAIN_WIDTH.set(40);
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        let new_width = get_swapchain_width();
        assert_ne!(old_width, new_width);

        assert_eq!(f.wait_get_poses(), None);
        SWAPCHAIN_WIDTH.set(20);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        let newer_width = get_swapchain_width();
        assert_eq!(newer_width, new_width);
    }

    #[test]
    fn get_frame_timing() {
        let f = Fixture::new();
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);

        let mut timing = MaybeUninit::new(vr::Compositor_FrameTiming::default());
        unsafe {
            (&raw mut (*timing.as_mut_ptr()).m_nSize)
                .write(std::mem::size_of::<vr::Compositor_FrameTiming>() as u32);
        }
        assert!(f.comp.GetFrameTiming(timing.as_mut_ptr(), 1));
        let small_size = std::mem::offset_of!(vr::Compositor_FrameTiming, m_HmdPose)
            + std::mem::size_of::<vr::TrackedDevicePose_t>();
        unsafe {
            (&raw mut (*timing.as_mut_ptr()).m_nSize).write(small_size as u32);
        }
        assert!(f.comp.GetFrameTiming(timing.as_mut_ptr(), 1));
        unsafe {
            (&raw mut (*timing.as_mut_ptr()).m_nSize).write(0);
        }
        assert!(!f.comp.GetFrameTiming(timing.as_mut_ptr(), 1));
    }

    #[test]
    fn zero_dims_texture() {
        let f = Fixture::new();
        SWAPCHAIN_WIDTH.set(0);
        SWAPCHAIN_HEIGHT.set(0);

        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);

        assert_eq!(f.wait_get_poses(), None);
        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(ctrl.swapchain_data.is_none());
            assert!(!ctrl.should_render);
        }
        SWAPCHAIN_WIDTH.set(10);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);

        assert_eq!(f.wait_get_poses(), None);
        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(ctrl.swapchain_data.is_none());
            assert!(ctrl.should_render);
        }
        SWAPCHAIN_HEIGHT.set(10);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);

        assert_eq!(f.wait_get_poses(), None);
        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(ctrl.swapchain_data.is_some());
            assert!(ctrl.should_render);
        }
    }

    #[test]
    fn vulkan_extensions() {
        let f = Fixture::new();

        fn tst(func: impl Fn(*mut c_char, u32) -> u32, dbg: &str) {
            // Normal flow
            let size = func(std::ptr::null_mut(), 0);
            let mut exts = vec![0; size as usize];
            func(exts.as_mut_ptr(), exts.len() as u32);

            let data = unsafe { CStr::from_ptr(exts.as_ptr()) };
            assert_eq!(data, c"VK_foo VK_bar", "{dbg}");

            // Oversized buffer
            let mut exts = vec![0; size as usize * 2];
            func(exts.as_mut_ptr(), exts.len() as u32);

            let data = unsafe { CStr::from_ptr(exts.as_ptr()) };
            assert_eq!(data, c"VK_foo VK_bar", "{dbg}");

            // Undersized buffer - should not crash
            let mut exts = vec![0];
            func(exts.as_mut_ptr(), exts.len() as u32);
        }

        tst(
            |buf, size| f.comp.GetVulkanInstanceExtensionsRequired(buf, size),
            "instance exts",
        );
        tst(
            |buf, size| {
                f.comp
                    .GetVulkanDeviceExtensionsRequired(std::ptr::null_mut(), buf, size)
            },
            "device exts",
        );
    }

    #[test]
    fn unsupported_format() {
        let f = Fixture::new();
        SWAPCHAIN_FORMAT.set(1);
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        let data = f.comp.openxr.session_data.get();
        let lock = data.comp_data.0.lock().unwrap();
        let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
            panic!("Frame controller was not set up or not faked!");
        };
        let data = ctrl
            .swapchain_data
            .as_ref()
            .expect("Swapchain data is missing");
        assert_eq!(data.initial_format, 1);
        assert_eq!(data.info.format, 0);
    }

    #[test]
    fn explicit_timing() {
        let f = Fixture::new();
        f.ensure_real_session(false);

        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        assert_eq!(f.wait_get_poses(), None);
        f.check_frame_state(fakexr::FrameState::Waited);

        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        f.check_frame_state(fakexr::FrameState::Begun);

        assert_eq!(f.submit(vr::EVREye::Left), None);
        f.check_frame_state(fakexr::FrameState::Begun);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        f.check_frame_state(fakexr::FrameState::Begun);

        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);
    }

    #[test]
    fn explicit_timing_no_submit() {
        let f = Fixture::new();
        f.ensure_real_session(false);

        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);
    }

    #[test]
    fn explicit_timing_multiple_waitgetposes() {
        let f = Fixture::new();
        f.ensure_real_session(false);
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );

        assert_eq!(f.wait_get_poses(), None);
        f.check_frame_state(fakexr::FrameState::Waited);
        assert_eq!(f.wait_get_poses(), None);
        f.check_frame_state(fakexr::FrameState::Waited);
    }

    #[test]
    fn explicit_timing_multiple_submit_explicit_timing_data() {
        let f = Fixture::new();
        f.ensure_real_session(false);
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );

        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        f.check_frame_state(fakexr::FrameState::Begun);
        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        f.check_frame_state(fakexr::FrameState::Begun);
    }

    #[test]
    fn explicit_timing_session_restart_after_waitgetposes() {
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );

        f.ensure_real_session(true);
        f.comp.openxr.restart_session();
        assert_eq!(f.wait_get_poses(), None);

        f.check_frame_state(fakexr::FrameState::Waited);
    }

    #[test]
    fn explicit_timing_submit_without_begun_frame() {
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );

        // The first Submit restarts the session while no frame has been begun, so the new swapchain has no acquired image when both eyes have been submitted
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);

        // The next frame should proceed as usual
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        f.check_frame_state(fakexr::FrameState::Begun);
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);
    }

    #[test]
    fn explicit_timing_unfocused() {
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );

        f.comp.SubmitExplicitTimingData();
        assert_eq!(f.submit(vr::EVREye::Left), DoNotHaveFocus);
        assert_eq!(f.submit(vr::EVREye::Right), DoNotHaveFocus);

        assert_eq!(f.wait_get_poses(), None);
        f.comp.SubmitExplicitTimingData();
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
    }

    #[test]
    fn submit_overlay_without_projection_layer() {
        use crate::overlay::OverlayMan;
        use vr::IVROverlay027_Interface;

        let f = Fixture::new();
        let overlays = Arc::new(OverlayMan::new(f.comp.openxr.clone(), &Injector::default()));
        f.comp.overlays.set(Arc::downgrade(&overlays));
        overlays.compositor.set(Arc::downgrade(&f.comp));

        let mut overlay = 0;
        assert_eq!(
            overlays.CreateOverlay(
                c"test_overlay".as_ptr(),
                c"TestOverlay".as_ptr(),
                &mut overlay
            ),
            vr::EVROverlayError::None
        );

        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(
            overlays.SetOverlayTexture(overlay, &FakeGraphicsData::texture(&f.vk)),
            vr::EVROverlayError::None
        );
        f.check_frame_state(fakexr::FrameState::Begun);
        assert_eq!(overlays.ShowOverlay(overlay), vr::EVROverlayError::None);
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);
    }

    /// Without a depth feed present, every end_frame stays plain: nothing
    /// arms, nothing pins, and the frame still submits color-only. Note
    /// fakexr does not advertise the depth extension, so this exercises the
    /// ext-off early return; the ext-on read-miss bucketing path is covered
    /// by the wire-level tests plus live heartbeat review.
    #[test]
    fn depth_stays_plain_without_feed() {
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        f.ensure_real_session(true);
        f.check_frame_state(fakexr::FrameState::Ended);

        let data = f.comp.openxr.session_data.get();
        let lock = data.comp_data.0.lock().unwrap();
        let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
            panic!("Frame controller was not set up or not faked!");
        };
        assert_eq!(ctrl.hb_depth, 0, "nothing armed without a feed");
        assert_eq!(ctrl.last_armed_gen, 0);
        assert!(ctrl.pinned_slot.is_none(), "nothing pinned without a feed");
        assert!(ctrl.depth_swapchain.is_none());
        assert!(ctrl.hb_frames >= 1, "heartbeat counts the frame");
    }

    /// Kill-switch off: no feed handle is opened and frames stay plain.
    /// Serialized through the depth env guard (process-global env); env is
    /// restored by a Drop guard so a mid-test panic can't leak `=0` into
    /// other tests.
    #[test]
    fn depth_off_with_kill_switch() {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
            }
        }
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "0");
        let _restore = Restore;
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        f.ensure_real_session(true);

        let data = f.comp.openxr.session_data.get();
        let lock = data.comp_data.0.lock().unwrap();
        let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
            panic!("Frame controller was not set up or not faked!");
        };
        assert!(ctrl.depth_feed.is_none(), "kill-switch opens no feed");
        assert_eq!(ctrl.hb_depth, 0);
        assert_eq!(ctrl.last_armed_gen, 0);
    }

    /// Fake backend (and Vulkan) carry no depth via trait defaults, which is
    /// what this pins: capability gate fails before any SHM work. (It does
    /// not prove a present feed would stay plain — that needs a
    /// depth-capable seam plus SHM, still open.)
    #[test]
    fn depth_unsupported_on_fake_backend() {
        use crate::graphics_backends::GraphicsBackend;
        assert!(!FakeGraphicsData::supports_depth());
        let f = Fixture::new();
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        f.ensure_real_session(true);

        let data = f.comp.openxr.session_data.get();
        let lock = data.comp_data.0.lock().unwrap();
        let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
            panic!("Frame controller was not set up or not faked!");
        };
        assert!(!ctrl.backend.depth_fd_import_viable());
        assert!(!ctrl.backend.depth_context_current());
        assert_eq!(ctrl.hb_depth, 0);
    }

    /// Heartbeat formats a fabricated window and resets every counter.
    /// No SHM/GL/runtime needed: same-module access drives the method
    /// directly with a backdated window start.
    #[test]
    fn depth_window_period_clamps() {
        // Non-positive falls back; sane values pass through; absurd values
        // clamp (huge must not widen the gate into stale accepts).
        assert_eq!(depth_window_period(0), DEPTH_FALLBACK_PERIOD_NS);
        assert_eq!(depth_window_period(-7), DEPTH_FALLBACK_PERIOD_NS);
        assert_eq!(depth_window_period(11_111_111), 11_111_111);
        assert_eq!(depth_window_period(8_333_333), 8_333_333);
        assert_eq!(depth_window_period(1), 4_000_000);
        assert_eq!(depth_window_period(3_999_999), 4_000_000);
        assert_eq!(depth_window_period(4_000_000), 4_000_000);
        assert_eq!(depth_window_period(50_000_000), 50_000_000);
        assert_eq!(depth_window_period(50_000_001), 50_000_000);
        assert_eq!(depth_window_period(i64::MAX), 50_000_000);
    }

    #[test]
    fn depth_heartbeat_formats_and_resets_window() {
        let f = Fixture::new();
        f.ensure_real_session(false);

        let data = f.comp.openxr.session_data.get();
        let mut lock = data.comp_data.0.lock().unwrap();
        let DynFrameController::Fake(ctrl) = lock.as_mut().unwrap() else {
            panic!("Frame controller was not set up or not faked!");
        };
        ctrl.hb_frames = 100;
        ctrl.hb_depth = 60;
        ctrl.hb_read_absent = 5;
        ctrl.hb_read_size = 1;
        ctrl.hb_read_torn = 2;
        ctrl.hb_read_magic = 1;
        ctrl.hb_read_version = 1;
        ctrl.hb_nonviable = 3;
        ctrl.hb_time_zero = 1;
        ctrl.hb_repeat = 4;
        ctrl.hb_alloc = 1;
        ctrl.hb_clip = 1;
        ctrl.hb_window = 6;
        ctrl.hb_dims = 1;
        ctrl.hb_flip = 1;
        ctrl.hb_no_context = 1;
        ctrl.hb_acquire_fail = 1;
        ctrl.hb_acquire_timeout = 2;
        ctrl.hb_poll_timeout = 1;
        ctrl.hb_poll_dead = 1;
        ctrl.hb_unsync = 7;
        ctrl.hb_copy_fail = 3;
        ctrl.hb_pin_fail = 2;
        ctrl.hb_pin_held = 1;
        ctrl.hb_swapchain = 1;
        ctrl.hb_epoch_raced = 1;
        ctrl.hb_end_fail = 1;
        ctrl.hb_depth_swapchains = 2;
        // Non-zero header identity: proves the seen[] format args flow and
        // that the heartbeat never resets them (state, not counters).
        ctrl.last_seen_gen = 42;
        ctrl.last_seen_epoch = 7;
        ctrl.last_seen_w = 20;
        ctrl.last_seen_h = 10;
        ctrl.age_stats.sample(1_000_000);
        ctrl.age_stats.sample(-500_000);
        ctrl.sample_acquire_ns(2_000_000);
        ctrl.sample_acquire_ns(4_000_000);
        ctrl.sample_poll_ns(500_000);
        // Backdate past the interval so the line fires (and would panic
        // the app thread on any format/args mismatch or div-zero).
        ctrl.hb_start = Instant::now() - Duration::from_secs(6);
        ctrl.maybe_log_depth_heartbeat();
        assert_eq!(ctrl.hb_frames, 0);
        assert_eq!(ctrl.hb_depth, 0);
        assert_eq!(ctrl.hb_read_absent, 0);
        assert_eq!(ctrl.hb_read_size, 0);
        assert_eq!(ctrl.hb_read_torn, 0);
        assert_eq!(ctrl.hb_read_magic, 0);
        assert_eq!(ctrl.hb_read_version, 0);
        assert_eq!(ctrl.hb_nonviable, 0);
        assert_eq!(ctrl.hb_time_zero, 0);
        assert_eq!(ctrl.hb_repeat, 0);
        assert_eq!(ctrl.hb_alloc, 0);
        assert_eq!(ctrl.hb_clip, 0);
        assert_eq!(ctrl.hb_window, 0);
        assert_eq!(ctrl.hb_dims, 0);
        assert_eq!(ctrl.hb_flip, 0);
        assert_eq!(ctrl.hb_no_context, 0);
        assert_eq!(ctrl.hb_acquire_fail, 0);
        assert_eq!(ctrl.hb_acquire_timeout, 0);
        assert_eq!(ctrl.hb_poll_timeout, 0);
        assert_eq!(ctrl.hb_poll_dead, 0);
        assert_eq!(ctrl.hb_unsync, 0);
        assert_eq!(ctrl.hb_copy_fail, 0);
        assert_eq!(ctrl.hb_pin_fail, 0);
        assert_eq!(ctrl.hb_pin_held, 0);
        assert_eq!(ctrl.hb_swapchain, 0);
        assert_eq!(ctrl.hb_epoch_raced, 0);
        assert_eq!(ctrl.hb_end_fail, 0);
        assert_eq!(ctrl.hb_depth_swapchains, 0);
        assert_eq!(
            (
                ctrl.last_seen_gen,
                ctrl.last_seen_epoch,
                ctrl.last_seen_w,
                ctrl.last_seen_h
            ),
            (42, 7, 20, 10),
            "seen[] identity survives the heartbeat (never reset)"
        );
        assert_eq!(ctrl.age_stats.count(), 0);
        assert_eq!(ctrl.acq_count, 0);
        assert_eq!(ctrl.acq_sum_ns, 0);
        assert_eq!(ctrl.acq_max_ns, 0);
        assert_eq!(ctrl.poll_count, 0);
        assert_eq!(ctrl.poll_sum_ns, 0);
        assert_eq!(ctrl.poll_max_ns, 0);
        // Immediate re-call is throttled: window counters survive.
        ctrl.hb_frames = 7;
        ctrl.maybe_log_depth_heartbeat();
        assert_eq!(ctrl.hb_frames, 7);
    }
}
