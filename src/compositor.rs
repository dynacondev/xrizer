use crate::{
    AtomicF64,
    clientcore::{Injected, Injector},
    depth_feed::{self, DepthControl, DepthFeed, DepthFrame, MODE_FD, SHM_VERSION_V2},
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
use std::time::Instant;
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
                fn take<G: GraphicsBackend>(ctrl: FrameController<G>) -> AnyTempBackendData
                where
                    AnyTempBackendData: From<TempBackendData<G>>,
                {
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

        // Propagate the depth-extension probe to the new controller.
        let depth_on = self.openxr.enabled_extensions.khr_composition_layer_depth;
        #[macros::any_graphics(DynFrameController)]
        fn set_depth<G: GraphicsBackend + 'static>(ctrl: &mut FrameController<G>, on: bool) {
            ctrl.set_depth_extension_enabled(on);
        }
        if let Some(ctrl) = session_data.comp_data.0.lock().unwrap().as_mut() {
            ctrl.with_any_graphics_mut::<set_depth>(depth_on);
        }

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
            overlays: Option<&OverlayMan>,
        ) where
            for<'b> &'b crate::overlay::AnySwapchainMap:
                TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
            <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
        {
            ctrl.end_frame(session_data, system, display_time, overlays)
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
        let overlays = self.overlays.get();

        ctrl.with_any_graphics_mut::<end_frame>((
            &session_data,
            &system,
            display_time,
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
            <G::Api as xr::Graphics>::Format: Eq + Clone + std::fmt::Debug,
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

struct SwapchainData<G: xr::Graphics> {
    swapchain: xr::Swapchain<G>,
    info: xr::SwapchainCreateInfo<G>,
    initial_format: G::Format,
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
    depth_ready_this_frame: bool,
    /// Resolved (near, far) for the armed depth upload (header wins, env
    /// fallback). Set alongside `depth_ready_this_frame`.
    depth_near_far: Option<(f32, f32)>,
    /// Per depth-swapchain-image generation already resident (PIXELS path
    /// only). The SHM feed updates at dump-tick rate while submits run per
    /// frame: re-uploading the same 51 MB every submit costs ~12 ms on the
    /// app thread. A slot holding this generation is still valid (we are
    /// the only writer; the runtime only reads), so skip the TexSubImage
    /// and just acquire/release to satisfy the runtime. Resized/reset
    /// whenever the depth swapchain is (re)created.
    depth_slot_gen: Vec<u64>,
    frame_seq: u64,
    frame_start_mono_ns: u64,
    /// Submit-rate heartbeat state: frames submitted since the last log
    /// line + window start. Answers "how fast is the app submitting?"
    /// end to end (server heartbeats cover the receive side). Throttled.
    submit_log_count: u64,
    submit_log_start_mono_ns: u64,
    depth_extension_enabled: bool,
    depth_off_logged: bool,
    /// One-shot stall warning: depth enabled but no fresh frame could be
    /// armed (SHM stale/dead, FD import failing, ...). Reset on every
    /// successful arm so flaps re-log instead of spamming.
    depth_stall_warned: bool,
}
supported_backends_enum!(enum DynFrameController: FrameController);

impl<G: GraphicsBackend> FrameController<G> {
    /// Creates the color swapchain, or `None` (with a warning) when the
    /// runtime refuses — e.g. a lost session. Callers keep the previous
    /// swapchain and degrade to dropped frames instead of aborting.
    fn init_swapchain(
        session_data: &SessionData,
        create_info: &mut xr::SwapchainCreateInfo<G::Api>,
        backend: &mut G,
    ) -> Option<(xr::Swapchain<G::Api>, <G::Api as xr::Graphics>::Format)>
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

        let swapchain = match session_data.create_swapchain(create_info) {
            Ok(s) => s,
            Err(err) => {
                warn!(
                    "Failed to create swapchain: {err} (info: {:#?})",
                    [
                        ("create_flags", format!("{:?}", create_info.create_flags)),
                        ("width", create_info.width.to_string()),
                        ("height", create_info.height.to_string()),
                        ("sample_count", create_info.sample_count.to_string())
                    ]
                );
                return None;
            }
        };

        let images = match swapchain.enumerate_images() {
            Ok(images) => images,
            Err(e) => {
                warn!("Failed to enumerate swapchain images ({e})");
                return None;
            }
        };

        backend.store_swapchain_images(images, create_info.format);
        debug!(
            "Created new swapchain: {}x{}, format = {:?}",
            create_info.width, create_info.height, create_info.format
        );

        Some((swapchain, initial_format))
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
            is_valid_swapchain_info(&info)
                .then(|| {
                    Self::init_swapchain(session_data, &mut info, &mut backend).map(
                        |(swapchain, initial_format)| SwapchainData {
                            swapchain,
                            info,
                            initial_format,
                        },
                    )
                })
                .flatten()
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
            depth_ready_this_frame: false,
            depth_near_far: None,
            depth_slot_gen: Vec::new(),
            frame_seq: 0,
            frame_start_mono_ns: 0,
            submit_log_count: 0,
            submit_log_start_mono_ns: 0,
            depth_extension_enabled: false,
            depth_off_logged: false,
            depth_stall_warned: false,
        }
    }

    fn set_depth_extension_enabled(&mut self, enabled: bool) {
        self.depth_extension_enabled = enabled;
    }

    /// Recreate on dims/format change. Returns whether a usable swapchain
    /// is now present. On failure the previous swapchain is kept and the
    /// frame degrades (stale image / failed end) instead of aborting the
    /// app.
    fn recreate_swapchain(
        &mut self,
        session_data: &SessionData,
        mut create_info: xr::SwapchainCreateInfo<G::Api>,
    ) -> bool
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + std::fmt::Debug,
    {
        let Some((swapchain, initial_format)) =
            Self::init_swapchain(session_data, &mut create_info, &mut self.backend)
        else {
            return false;
        };

        self.swapchain_data = Some(SwapchainData {
            swapchain,
            info: create_info,
            initial_format,
        });
        // Depth dims derive from color dims; drop it so end_frame recreates
        // at the new size (GL session only; other backends never create it).
        self.depth_swapchain = None;
        self.depth_slot_gen = Vec::new();
        self.depth_ready_this_frame = false;
        self.depth_near_far = None;
        self.acquire_swapchain_image();
        self.eyes_submitted = Default::default();
        true
    }

    /// Acquire + wait a color swapchain image. Returns false (leaving
    /// `image_acquired` unset) when the runtime is gone: callers degrade to
    /// dropped frames instead of aborting the app process.
    fn acquire_swapchain_image(&mut self) -> bool {
        let swapchain = &mut self
            .swapchain_data
            .as_mut()
            .expect("Can't acquire swapchain image with no swapchain!")
            .swapchain;

        let idx = match swapchain.acquire_image() {
            Ok(i) => i as usize,
            Err(e) => {
                warn!("failed to acquire swapchain image ({e}); dropping frame");
                return false;
            }
        };
        self.image_index = idx;

        trace!("waiting image");
        {
            tracy_span!("wait swapchain image");
            if let Err(e) = swapchain.wait_image(xr::Duration::INFINITE) {
                warn!("failed to wait for swapchain image ({e}); dropping frame");
                return false;
            }
        }

        self.image_acquired = true;
        true
    }

    fn wait_frame(&mut self) -> (xr::Time, i64) {
        let frame_state = {
            tracy_span!("wait frame");
            match self.waiter.wait() {
                Ok(s) => s,
                Err(e) => {
                    warn!("xrWaitFrame failed ({e}); pausing rendering");
                    self.should_render = false;
                    // Dead session: report a dummy tick; the runtime is
                    // gone and every downstream OpenXR call will fail
                    // gracefully from here.
                    return (xr::Time::from_nanos(0), 11_111_111);
                }
            }
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
            if let Err(e) = self
                .swapchain_data
                .as_mut()
                .expect("Image is acquired, yet we have no swapchain?")
                .swapchain
                .release_image()
            {
                warn!("failed to release old swapchain image ({e})");
            }
        }

        if self.swapchain_data.is_some() {
            self.acquire_swapchain_image();
        }

        {
            tracy_span!("begin frame");
            if let Err(e) = self.stream.begin() {
                warn!("failed to begin frame ({e}); skipping frame start");
                self.should_render = false;
                return;
            }
        }
        self.eyes_submitted = [None; 2];
        self.submitting_null = false;
        self.depth_ready_this_frame = false;
        self.depth_near_far = None;
        self.frame_start_mono_ns = depth_feed::now_mono_ns();
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
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
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
                    if !self.swapchain_data.as_ref().is_some_and(|data| {
                        is_usable_swapchain(&data.info, data.initial_format, &new_info)
                    }) {
                        info!("recreating swapchain (for {eye:?})");
                        // Recreate can fail on a lost session: fall back to
                        // a null submit for this eye rather than indexing
                        // into a missing/stale swapchain image list.
                        // Checked before the mix assert so both eyes can
                        // degrade together.
                        if !self.recreate_swapchain(session_data, new_info) {
                            warn!("no usable swapchain (for {eye:?}); null-submitting");
                            return None;
                        }
                    }

                    assert!(
                        !self.submitting_null,
                        "App submitted a null texture and a normal texture in the same frame"
                    );

                    Some(SubmittedEye {
                        extent: self.backend.copy_texture_to_swapchain(
                            eye,
                            texture,
                            color_space,
                            bounds,
                            self.image_index,
                            submit_flags,
                        ),
                        flip_vertically: bounds.vertically_flipped(),
                    })
                })
                .flatten()
                .or_else(|| {
                    trace!("submitting null this frame");
                    self.submitting_null = true;
                    Some(Default::default())
                })
        } else {
            Some(Default::default())
        };

        // Depth upload rides the Submit GL context (proven current by the
        // color copy above). The control header decides the mode: FD (GPU
        // import + copy, timeline association) or PIXELS (SHM upload,
        // timestamp association, unchanged behavior).
        if self.should_render
            && !self.submitting_null
            && !self.depth_ready_this_frame
            && G::supports_depth()
            && self.depth_extension_enabled
            && depth_feed::depth_master_enabled()
        {
            if self.depth_feed.is_none() {
                self.depth_feed = DepthFeed::open();
            }
            match self.depth_feed.as_ref().and_then(|f| f.read_control()) {
                Some(ctl)
                    if ctl.version == SHM_VERSION_V2
                        && ctl.mode == MODE_FD
                        && self.backend.depth_fd_import_viable() =>
                {
                    self.try_upload_depth_fd(session_data, &ctl);
                }
                other => self.try_upload_depth(session_data, other.as_ref()),
            }
        }

        trace!("submitted {eye:?}");
        if self.eyes_submitted.iter().all(|eye| eye.is_some()) {
            // If the session was restarted while no frame was begun, nothing has been acquired from the new swapchain yet, so there is nothing to release
            if self.image_acquired
                && let Some(data) = self.swapchain_data.as_mut()
            {
                trace!("releasing image");
                // A lost session surfaces here; never abort the app for it.
                if let Err(e) = data.swapchain.release_image() {
                    warn!("failed to release swapchain image ({e})");
                }
            }
            self.image_acquired = false;
        }

        Ok(())
    }

    /// SHM poll + depth swapchain acquire/upload/release for this frame.
    /// Called from `submit_impl` where the app GL context is current.
    /// Uploads both halves atomically; arms `depth_ready_this_frame` only
    /// on full success so end_frame chains both eyes or neither.
    ///
    /// `ctl` is the already-read control header (PIXELS mode or no FD
    /// viability). When it carries the generation already resident in the
    /// acquired slot, the 51 MB SHM pixel copy is skipped: acquire/release
    /// still runs to satisfy the runtime and depth still arms.
    fn try_upload_depth(&mut self, session_data: &SessionData, ctl: Option<&DepthControl>)
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        if self.depth_feed.is_none() {
            self.depth_feed = DepthFeed::open();
        }
        // Re-check SHM w/h every read: the producer recreates the segment
        // mid-run on frame-graph rebuilds; skip frames on mismatch.
        let (cw, ch) = match self.swapchain_data.as_ref() {
            Some(c) => (c.info.width, c.info.height),
            None => return,
        };
        if cw == 0 {
            return;
        }
        // Fast path: the control header alone proves the resident data is
        // current — no SHM pixel-plane copy needed. Dims and format still
        // gate on the header so a rebuild that changes geometry can never
        // arm stale depth. On a slot miss the image is released back and
        // the full path below re-acquires cleanly.
        if let Some(ctl) = ctl
            && ctl.mode == depth_feed::MODE_PIXELS
            && ctl.generation != 0
            && ctl.eye_layout == depth_feed::EYE_SIDEBYSIDE
            && ctl.w == cw * 2
            && ctl.h == ch
            && self.ensure_depth_swapchain(session_data, ctl.w / 2, ctl.h, ctl.format)
        {
            let acquired: Option<usize> = (|| {
                let data = self.depth_swapchain.as_mut()?;
                let idx = data.swapchain.acquire_image().ok()? as usize;
                data.swapchain.wait_image(xr::Duration::INFINITE).ok()?;
                Some(idx)
            })();
            if let Some(idx) = acquired {
                if self.depth_slot_gen.get(idx) == Some(&ctl.generation) {
                    if let Some(data) = self.depth_swapchain.as_mut() {
                        let _ = data.swapchain.release_image();
                    }
                    self.depth_near_far =
                        Some(depth_feed::resolve_near_far(ctl.near_z, ctl.far_z));
                    self.depth_ready_this_frame = true;
                    self.depth_stall_warned = false;
                    return;
                }
                if let Some(data) = self.depth_swapchain.as_mut() {
                    let _ = data.swapchain.release_image();
                }
            }
        }
        let frame: DepthFrame = match (|| {
            let feed = self.depth_feed.as_ref()?;
            let now = if self.frame_start_mono_ns != 0 {
                self.frame_start_mono_ns
            } else {
                depth_feed::now_mono_ns()
            };
            feed.latest_for(now, depth_feed::max_age_ms())
        })() {
            Some(f) => f,
            None => {
                // Feed enabled but nothing fresh: producer stopped,
                // SHM dead/stale, or frames arriving older than max age.
                // Warn once per episode (reset on every successful arm).
                if self.depth_feed.is_some() && !self.depth_stall_warned {
                    self.depth_stall_warned = true;
                    warn!("depth enabled but no fresh frame (SHM stale, dead, or producer stopped?)");
                }
                return;
            }
        };
        // Re-check SHM w/h every read: the producer recreates the segment
        // mid-run on frame-graph rebuilds; skip frames on mismatch.
        let (cw, ch) = match self.swapchain_data.as_ref() {
            Some(c) => (c.info.width, c.info.height),
            None => return,
        };
        if cw == 0 || frame.w != cw * 2 || frame.h != ch {
            debug!(
                "depth dims {}x{} vs color {cw}x{ch} — plain views this frame",
                frame.w, frame.h,
            );
            return;
        }
        let eye_w = frame.w / 2;
        let eye_h = frame.h;
        if !self.ensure_depth_swapchain(session_data, eye_w, eye_h, frame.vk_format) {
            return;
        }
        let idx: usize = match (|| {
            let data = self.depth_swapchain.as_mut()?;
            let idx = data.swapchain.acquire_image().ok()? as usize;
            data.swapchain.wait_image(xr::Duration::INFINITE).ok()?;
            Some(idx)
        })() {
            Some(i) => i,
            None => return,
        };
        // Same generation already resident in this slot (we are the only
        // writer): skip the ~12 ms TexSubImage, just release + arm. The
        // runtime samples whatever we last released, so every frame still
        // chains valid depth.
        if self.depth_slot_gen.get(idx) == Some(&frame.present_no) {
            if let Some(data) = self.depth_swapchain.as_mut() {
                let _ = data.swapchain.release_image();
            }
            self.depth_near_far = Some(depth_feed::resolve_near_far(frame.near_z, frame.far_z));
            self.depth_ready_this_frame = true;
            self.depth_stall_warned = false;
            return;
        }
        let t0 = Instant::now();
        let ok = self.backend.upload_depth_images(idx, &frame, eye_w, eye_h);
        debug!(
            "depth upload {:.2} ms ({}x{}x2, frame {}, present {})",
            t0.elapsed().as_secs_f64() * 1000.0,
            eye_w,
            eye_h,
            self.frame_seq,
            frame.present_no,
        );
        // Release before end (color path releases before end too).
        if let Some(data) = self.depth_swapchain.as_mut() {
            let _ = data.swapchain.release_image();
        }
        if ok {
            if let Some(g) = self.depth_slot_gen.get_mut(idx) {
                *g = frame.present_no;
            }
            self.depth_near_far = Some(depth_feed::resolve_near_far(frame.near_z, frame.far_z));
            self.depth_ready_this_frame = true;
            self.depth_stall_warned = false;
        }
    }

    /// FD-mode upload: GPU-side copy from the imported dma-buf ring.
    /// Runs in `submit_impl` (GL context current). No CPU upload, no
    /// timestamp matching — the per-slot binary semaphore is the timeline.
    /// The ack (`consumed_gen`) is stored only for fully-copied generations
    /// (fence-drained), so the producer never reuses an in-flight slot.
    fn try_upload_depth_fd(&mut self, session_data: &SessionData, ctl: &DepthControl)
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        use depth_feed::{FORMAT_D32F, FORMAT_D32S8};
        if ctl.generation == 0 {
            return;
        }
        if ctl.format != FORMAT_D32F && ctl.format != FORMAT_D32S8 {
            return;
        }
        if ctl.eye_layout != depth_feed::EYE_SIDEBYSIDE || ctl.w % 2 != 0 {
            return;
        }
        // Dims are authoritative per read: never stretch, skip the frame.
        let (cw, ch) = match self.swapchain_data.as_ref() {
            Some(c) => (c.info.width, c.info.height),
            None => return,
        };
        if cw == 0 || ctl.w != cw * 2 || ctl.h != ch {
            debug!(
                "depth dims {}x{} vs color {cw}x{ch} — plain views this frame",
                ctl.w, ctl.h,
            );
            return;
        }
        let eye_w = ctl.w / 2;
        let eye_h = ctl.h;
        if !self.ensure_depth_swapchain(session_data, eye_w, eye_h, ctl.format) {
            return;
        }
        let idx: usize = match (|| {
            let data = self.depth_swapchain.as_mut()?;
            let idx = data.swapchain.acquire_image().ok()? as usize;
            data.swapchain.wait_image(xr::Duration::INFINITE).ok()?;
            Some(idx)
        })() {
            Some(i) => i,
            None => return,
        };
        let t0 = Instant::now();
        let res = self.backend.import_and_copy_depth_fd(&DepthFdRequest {
            setup_epoch: ctl.setup_epoch,
            seg_len: ctl.seg_len,
            w: ctl.w,
            h: ctl.h,
            vk_format: ctl.format,
            mem_fds: ctl.mem_fds,
            sem_fds: ctl.sem_fds,
            generation: ctl.generation,
            depth_image_index: idx,
            eye_w,
            eye_h,
            alloc_bytes: ctl.alloc_bytes,
        });
        debug!(
            "depth FD copy {:.2} ms ({}x{}x2, frame {}, generation {})",
            t0.elapsed().as_secs_f64() * 1000.0,
            eye_w,
            eye_h,
            self.frame_seq,
            ctl.generation,
        );
        // Release before end (color path releases before end too).
        if let Some(data) = self.depth_swapchain.as_mut() {
            let _ = data.swapchain.release_image();
        }
        // Ack only fully-copied generations (fence-drained in the backend).
        // Skip the SHM write when the producer already observed it.
        if let Some(ack) = res.ack_gen
            && ack != ctl.consumed_gen
            && let Some(feed) = self.depth_feed.as_ref()
            && feed.store_consumed(ack)
        {
            debug!("depth acked generation {ack} (was {})", ctl.consumed_gen);
        }
        if res.copied {
            self.depth_near_far = Some(depth_feed::resolve_near_far(ctl.near_z, ctl.far_z));
            self.depth_ready_this_frame = true;
            self.depth_stall_warned = false;
        } else if !self.depth_stall_warned {
            self.depth_stall_warned = true;
            warn!("depth FD copy failed (generation {})", ctl.generation);
        }
    }

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
        if let Some(d) = self.depth_swapchain.as_ref() {
            if d.info.width == eye_w && d.info.height == eye_h && d.initial_format == want_fmt {
                return true;
            }
        }
        let formats = match session_data.swapchain_formats::<G>() {
            Some(f) => f,
            None => return false,
        };
        if !formats.contains(&want_fmt) {
            if !self.depth_off_logged {
                self.depth_off_logged = true;
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
            .depth_swapchain_create_info(color_info, want_fmt.clone())
        {
            Some(i) => i,
            None => return false,
        };
        // Enforce per-eye dims derived from the color swapchain.
        info.width = eye_w;
        info.height = eye_h;
        let usage_full = info.usage_flags;
        match session_data.create_swapchain(&info) {
            Ok(swapchain) => {
                let images = match swapchain.enumerate_images() {
                    Ok(images) => images,
                    Err(e) => {
                        warn!("failed to enumerate depth swapchain images ({e})");
                        return false;
                    }
                };
                self.depth_slot_gen = vec![0; images.len()];
                self.backend
                    .store_depth_swapchain_images(images, info.format);
                debug!(
                    "Created depth swapchain: {}x{}x2, format = {:?}, usage = {:?}",
                    info.width, info.height, info.format, usage_full,
                );
                self.depth_swapchain = Some(SwapchainData {
                    swapchain,
                    info,
                    initial_format: want_fmt,
                });
                true
            }
            Err(e) => {
                debug!(
                    "depth create with {usage_full:?} failed ({e}); retrying DEPTH_STENCIL only"
                );
                info.usage_flags = xr::SwapchainUsageFlags::DEPTH_STENCIL_ATTACHMENT;
                match session_data.create_swapchain(&info) {
                    Ok(swapchain) => {
                        let images = match swapchain.enumerate_images() {
                            Ok(images) => images,
                            Err(e) => {
                                warn!(
                                    "failed to enumerate depth swapchain images ({e})"
                                );
                                return false;
                            }
                        };
                        self.depth_slot_gen = vec![0; images.len()];
                        self.backend
                            .store_depth_swapchain_images(images, info.format);
                        info!(
                            "Created depth swapchain (fallback usage): {}x{}x2, format = {:?}",
                            info.width, info.height, info.format,
                        );
                        self.depth_swapchain = Some(SwapchainData {
                            swapchain,
                            info,
                            initial_format: want_fmt,
                        });
                        true
                    }
                    Err(e2) => {
                        warn!("depth swapchain creation failed ({e2}); color-only");
                        false
                    }
                }
            }
        }
    }

    /// Depth path for `end_frame`. Returns true when the frame was submitted
    /// with chained `CompositionLayerDepthInfoKHR` views. Returns false on any
    /// gate failure (extension off, master off, no ready upload, near == far)
    /// — caller then uses the plain path. Uploads happen in Submit (GL
    /// context current); both halves were uploaded atomically there, so this
    /// chains both eyes or neither and performs no GL work.
    fn try_end_frame_with_depth(
        &mut self,
        session_data: &SessionData,
        system: &System,
        display_time: xr::Time,
        overlays: Option<&OverlayMan>,
    ) -> bool
    where
        for<'b> &'b crate::overlay::AnySwapchainMap:
            TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        if !(self.should_render
            && !self.submitting_null
            && self.eyes_submitted.iter().all(|eye| eye.is_some()))
        {
            return false;
        }
        if !self.depth_extension_enabled || !G::supports_depth() {
            return false;
        }
        if !depth_feed::depth_master_enabled() {
            return false;
        }
        // Armed by try_upload_depth / try_upload_depth_fd during Submit.
        // Both halves were handled atomically there: both eyes or neither.
        if !self.depth_ready_this_frame {
            return false;
        }
        let Some((near_z, far_z)) = self.depth_near_far else {
            return false;
        };
        if near_z == far_z {
            warn!("depth near == far ({near_z}); plain views this frame");
            return false;
        }
        let (eye_w, eye_h) = match self.depth_swapchain.as_ref() {
            Some(d) => (d.info.width, d.info.height),
            None => return false,
        };
        if eye_w == 0 || eye_h == 0 {
            return false;
        }

        // Build chained views: safe wrappers from raw sys structs whose
        // `next` points at per-eye DepthInfo. All referents (swapchains,
        // depth_infos, views) outlive the synchronous end() call.
        let color_data = match self.swapchain_data.as_ref() {
            Some(c) => c,
            None => return false,
        };
        let depth_data = match self.depth_swapchain.as_ref() {
            Some(d) => d,
            None => return false,
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
                flip_vertically,
            } = eyes[eye_index]
                .unwrap_or_else(|| panic!("Eye {eye_index} has not been submitted!"));
            let mut fov = view.fov;
            if flip_vertically {
                std::mem::swap(&mut fov.angle_up, &mut fov.angle_down);
            }
            poses.push(pose);
            fovs.push(fov);
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
            return false;
        }

        let depth_infos: Vec<xr::sys::CompositionLayerDepthInfoKHR> = (0..2)
            .map(|i| xr::sys::CompositionLayerDepthInfoKHR {
                ty: xr::sys::CompositionLayerDepthInfoKHR::TYPE,
                next: std::ptr::null(),
                sub_image: *depth_subs[i].as_raw(),
                min_depth: 0.0,
                max_depth: 1.0,
                near_z,
                far_z,
            })
            .collect();
        // Monado rejects near == far and out-of-range min/max; checked above.
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
                true
            }
            Err(e) => {
                warn!("depth frame end failed ({e}); retrying plain views");
                false
            }
        }
    }

    fn end_frame(
        &mut self,
        session_data: &SessionData,
        system: &System,
        display_time: xr::Time,
        overlays: Option<&OverlayMan>,
    ) where
        for<'b> &'b crate::overlay::AnySwapchainMap:
            TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        self.frame_seq = self.frame_seq.wrapping_add(1);

        // Submit-rate heartbeat (logging only): frames submitted per
        // second end to end. Throttled; the submit path is untouched.
        {
            self.submit_log_count += 1;
            let now = depth_feed::now_mono_ns();
            if self.submit_log_start_mono_ns == 0 {
                self.submit_log_start_mono_ns = now;
            } else if now - self.submit_log_start_mono_ns > 5_000_000_000 {
                let dt = (now - self.submit_log_start_mono_ns) as f64 / 1e9;
                info!(
                    "submits {:.1}/s ({} frames)",
                    self.submit_log_count as f64 / dt,
                    self.submit_log_count
                );
                self.submit_log_count = 0;
                self.submit_log_start_mono_ns = now;
            }
        }

        // Depth path first; any gate failure falls through to the plain path.
        if self.try_end_frame_with_depth(session_data, system, display_time, overlays) {
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
            .inspect_err(|e| warn!("frame end failed ({e}); frame dropped"))
            .ok();

        trace!("frame submitted");
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

    #[test]
    fn depth_fd_mode_inert_without_import_viable() {
        // v2 FD segment present + master on, but the Fake backend reports
        // no import viability: frames go through plain views, no depth
        // state armed, no consumed ack written.
        let _lock = crate::depth_feed::env_guard();
        let seg_name = format!("/xrizer-depth-test-{}-fd", std::process::id());
        unsafe {
            let cname = std::ffi::CString::new(seg_name.clone()).unwrap();
            libc::shm_unlink(cname.as_ptr());
            let len = 128usize;
            let fd = libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            );
            assert!(fd >= 0);
            assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            assert_ne!(ptr, libc::MAP_FAILED);
            libc::close(fd);
            let hdr = crate::depth_feed::XdShmHeaderV2 {
                magic: 0x5850_4450,
                version: 2,
                seq: 1,
                w: 20,
                h: 10,
                format: 126,
                eye_layout: 1,
                present_no: 1,
                time_mono_ns: 0,
                min_depth: 0.0,
                max_depth: 1.0,
                _pad56: [0, 0],
                mode: 1,
                setup_epoch: 3,
                consumed_gen: 0,
                mem_fds: [-1, -1, -1],
                sem_fds: [-1, -1, -1],
                near_z: 0.5,
                far_z: 2000.0,
                reserved: [0; 4],
            };
            (ptr as *mut crate::depth_feed::XdShmHeaderV2).write_unaligned(hdr);
            (ptr as *mut u32).add(2).write_volatile(2);
            libc::munmap(ptr, len);
        }
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        crate::depth_feed::set_env_for_tests("XP_DEPTH_SHM_NAME", &seg_name);

        let f = Fixture::new();
        f.ensure_real_session(false);
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        {
            let data = f.comp.openxr.session_data.get();
            let mut frame_lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = frame_lock.as_mut().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            ctrl.should_render = true;
        }
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);

        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(!ctrl.depth_ready_this_frame);
            assert!(ctrl.depth_near_far.is_none());
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
        crate::depth_feed::del_env_for_tests("XP_DEPTH_SHM_NAME");
        unsafe {
            let cname = std::ffi::CString::new(seg_name).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }
    }

    #[test]
    fn depth_inert_with_unsupported_backend() {
        // Master switch on, but the Fake backend has no depth support:
        // frames must go through plain views with no depth state armed.
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        crate::depth_feed::set_env_for_tests(
            "XP_DEPTH_SHM_NAME",
            "/xrizer-depth-test-absent-tier2",
        );

        let f = Fixture::new();
        f.ensure_real_session(false);
        f.comp.SetExplicitTimingMode(
            vr::EVRCompositorTimingMode::Explicit_ApplicationPerformsPostPresentHandoff,
        );
        assert_eq!(f.wait_get_poses(), None);
        assert_eq!(f.comp.SubmitExplicitTimingData(), None);
        // Force the render path: fakexr never reports should_render.
        {
            let data = f.comp.openxr.session_data.get();
            let mut frame_lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = frame_lock.as_mut().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            ctrl.should_render = true;
        }
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);

        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(ctrl.depth_swapchain.is_none());
            assert!(ctrl.depth_feed.is_none());
            assert!(!ctrl.depth_ready_this_frame);
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
        crate::depth_feed::del_env_for_tests("XP_DEPTH_SHM_NAME");
    }
}
