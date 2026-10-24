use crate::{
    AtomicF64,
    clientcore::{Injected, Injector},
    depth_feed::{self, Backoff, DepthControl, DepthFeed},
    graphics_backends::{
        DepthFdRequest, GraphicsBackend, SupportedBackend, supported_backends_enum,
    },
    input::Input,
    openxr_data::{self, FrameStream, OpenXrData, SessionCreateInfo, SessionData},
    overlay::OverlayMan,
    system::System,
    tracy_span,
};

use log::{debug, error, info, trace, warn};
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
        // Trace-level frame pacing log (off unless RUST_LOG enables trace):
        // one line per WaitGetPoses-driven wait, with the predicted
        // display time handed to the app. Thread id separates the render
        // thread's waits from the pose-updater pump's: two interleaved
        // streams advancing display_time independently would prove
        // double-pumping. Compare its rate against the end_frame rate
        // in the same log: waits advancing faster than ends means poses
        // get predicted more than one frame ahead.
        trace!(
            "waited frame: display_time {:?} period {}ns tid {:?}",
            display_time, display_period,
            std::thread::current().id()
        );
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

        let old = session_data.comp_data.0.lock().unwrap().replace(
            backend_data.with_any_graphics_owned::<new_frame_controller>((
                session_data,
                waiter,
                stream,
            )),
        );
        // Texture-switch path: the old controller was never harvested into
        // tmp_backend, so replacing comp_data just dropped it — release its
        // backend GL imports here (they reference the old swapchain's
        // images). Normal restarts replace None: no-op. Deletes without a
        // current GL context are harmless no-ops; the new controller
        // reimports on demand (epoch-keyed).
        if let Some(mut old_ctrl) = old {
            #[macros::any_graphics(DynFrameController)]
            fn drop_imports<G: GraphicsBackend + 'static>(ctrl: &mut FrameController<G>) {
                // B3: release the pending depth image before dropping
                // backend GL imports (the index belongs to the old
                // swapchain, leaked otherwise).
                ctrl.release_depth_pending();
                ctrl.backend.drop_depth_imports();
            }
            old_ctrl.with_any_graphics_mut::<drop_imports>(());
        }

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
        ) -> xr::Result<()>
        where
            for<'b> &'b crate::overlay::AnySwapchainMap:
                TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
            for<'a> &'a openxr_data::GraphicalSession:
                TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
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

        // B2: exactly one stream.end per frame inside end_frame. A
        // single-end failure returns here with no second end (logged,
        // frame still advances — same as a plain-end attempt completing).
        if ctrl
            .with_any_graphics_mut::<end_frame>((
                &session_data,
                &system,
                display_time,
                overlays.as_deref(),
            ))
            .is_err()
        {
            warn!("end_frame failed; no plain retry (single end)");
        }

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
    /// Validated (near, far) for the armed depth upload (header-only, no
    /// xrizer-side defaults). Set alongside `depth_ready_this_frame`.
    depth_near_far: Option<(f32, f32)>,
    /// Header (min, max) mapping range for the armed upload. Sanitized to
    /// finite values at arm time (Monado rejects garbage with the frame).
    depth_min_max: Option<(f32, f32)>,
    /// Producer generation armed for this frame (present_no). Set alongside
    /// `depth_ready_this_frame`; the end_frame upgrade uses it to detect a
    /// newer present.
    depth_armed_generation: Option<u64>,
    /// Debug-only submit heartbeat: per-window submits with the
    /// depth-armed vs plain split. Throttled 5 s; the submit path is untouched.
    /// (The total is derived as armed + plain at emit time — every end takes
    /// exactly one of those branches — so no separate counter is stored.)
    submit_log_start_mono_ns: u64,
    depth_armed_frames: u64,
    depth_plain_frames: u64,
    /// Debug-only upgrade accounting (same heartbeat window): end_frame
    /// upgrades that copied a newer generation vs attempts that kept the
    /// submit-time arming (no context, stale, or failed copy — all
    /// silent by design). Answers live whether the 0-lag path executes.
    depth_upgrade_ok: u64,
    depth_upgrade_pass: u64,
    depth_extension_enabled: bool,
    depth_off_logged: bool,
    /// One-shot stall warning: SHM has no fresh frame (stale, dead, or
    /// producer stopped). Reset as soon as a fresh header passes the
    /// staleness gate, so the next stall re-warns.
    depth_stale_warned: bool,
    /// One-shot failure warning: near/far invalid, swapchain missing,
    /// acquire/wait timeout, copy failure, or depth submit rejection.
    /// Reset on every successful copy. Kept separate from the stale flag
    /// so one episode cannot mask the other.
    depth_fail_warned: bool,
    /// Last quiet-path (end_frame upgrade) skip reason, warned once per
    /// reason episode. A changed reason re-warns; success clears it.
    /// Lets the next live run name the ack-freeze cause (stale vs dims
    /// vs acquire vs copy) instead of going silent.
    depth_upgrade_skip_reason: Option<&'static str>,
    /// Last loud-path (submit) SHM-read skip reason (B1), warned once per
    /// reason episode via `note_submit_skip`. Torn/recreate races land
    /// here instead of silent-plain.
    depth_submit_skip_reason: Option<&'static str>,
    /// Loud-path submit skips since the last heartbeat (B1): SHM read
    /// misses that never reached the upload. Surfaced in the frames line.
    depth_submit_skips: u64,
    /// Armed header timestamp (`present_no`'s `time_mono_ns`) captured at
    /// arm time (B7). Re-validated at chain time so a 100ms-boundary flap
    /// between submit and end degrades to plain with a reason.
    depth_armed_time_ns: Option<u64>,
    /// Single depth image acquired per frame (B3): acquired at the
    /// submit-time copy, re-used by the end-frame upgrade for a newer
    /// generation, released exactly once at end of frame on every path.
    depth_pending_image: Option<usize>,
    /// Ack-store failures since the last heartbeat (B4): `store_consumed`
    /// returned false for a fully-copied generation. Warned once per
    /// episode; success clears.
    depth_ack_store_fails: u64,
    depth_ack_store_warned: bool,
    /// display_time of the previous end_frame: an end reusing the same
    /// timestamp re-presents an already-submitted frame (repeat). The
    /// frames line reports the per-window repeat count so the next live
    /// run shows whether this bridge ever re-emits (it shouldn't — the
    /// runtime's multi layer recommits on its own cadence).
    last_end_display_time: Option<xr::Time>,
    end_repeat_count: u64,
    /// Earliest retry time after a depth swapchain creation failure.
    /// Transient runtime errors cost skipped ticks, never stalls or spam.
    /// Shared [`Backoff`] helper (S3): same 2 s skip-until-Instant pattern
    /// as the FD import path; warns stay at the failure sites below.
    depth_swapchain_backoff: Backoff,
    /// Happy path emits exactly one info! total (first successful arm).
    depth_first_arm_logged: bool,
    /// Test-only acquire/release accounting (B3, no GL): incremented by
    /// `acquire_depth_image` / `release_depth_pending` so the Fake-backend
    /// test proves acquire==release==ends and one acquire per frame.
    #[cfg(test)]
    pub depth_test_acquires: u64,
    #[cfg(test)]
    pub depth_test_releases: u64,
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
            depth_ready_this_frame: false,
            depth_near_far: None,
            depth_min_max: None,
            depth_armed_generation: None,
            submit_log_start_mono_ns: 0,
            depth_armed_frames: 0,
            depth_plain_frames: 0,
            depth_upgrade_ok: 0,
            depth_upgrade_pass: 0,
            depth_extension_enabled: false,
            depth_off_logged: false,
            depth_stale_warned: false,
            depth_fail_warned: false,
            depth_upgrade_skip_reason: None,
            depth_submit_skip_reason: None,
            depth_submit_skips: 0,
            depth_armed_time_ns: None,
            depth_pending_image: None,
            depth_ack_store_fails: 0,
            depth_ack_store_warned: false,
            last_end_display_time: None,
            end_repeat_count: 0,
            depth_swapchain_backoff: Backoff::new(),
            depth_first_arm_logged: false,
            #[cfg(test)]
            depth_test_acquires: 0,
            #[cfg(test)]
            depth_test_releases: 0,
        }
    }

    fn set_depth_extension_enabled(&mut self, enabled: bool) {
        // Release any in-flight depth image before the swapchain state
        // changes (B3): the pending index belongs to the old swapchain.
        self.release_depth_pending();
        if !enabled {
            self.depth_swapchain = None;
            self.backend.drop_depth_imports();
            self.depth_ready_this_frame = false;
            self.depth_near_far = None;
            self.depth_min_max = None;
            self.depth_armed_generation = None;
            self.depth_armed_time_ns = None;
        }
        self.depth_extension_enabled = enabled;
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
        // Depth dims derive from color dims; drop it so it is recreated
        // at the new size (GL session only; other backends never create it).
        // Release the backend GL imports too: they reference the old
        // swapchain's images and would otherwise leak per resize.
        // B3: release the pending depth image BEFORE dropping the
        // swapchain it was acquired from.
        self.release_depth_pending();
        self.depth_swapchain = None;
        self.backend.drop_depth_imports();
        self.depth_ready_this_frame = false;
        self.depth_near_far = None;
        self.depth_min_max = None;
        self.depth_armed_generation = None;
        self.depth_armed_time_ns = None;
        self.acquire_swapchain_image();
        self.eyes_submitted = Default::default();
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
        // B3: a pending depth image must never survive into the next
        // frame. end_frame releases on every path, so this is normally
        // None; release defensively (e.g. a frame that never ended).
        self.release_depth_pending();
        self.depth_ready_this_frame = false;
        self.depth_near_far = None;
        self.depth_min_max = None;
        self.depth_armed_generation = None;
        self.depth_armed_time_ns = None;
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

        // Depth upload rides the Submit GL context (proven current by the
        // color copy above). FD mode only: GPU-side copy from the imported
        // dma-buf ring, no CPU upload. Arms `depth_ready_this_frame` only
        // on full success so end_frame chains both eyes or neither.
        // Render-state gates stay here (they differ from the upgrade path);
        // the shared feed/viability/read/upload tail lives in `poll_depth`.
        if self.should_render
            && !self.submitting_null
            && !self.depth_ready_this_frame
            && G::supports_depth()
            && self.depth_extension_enabled
            && depth_feed::depth_master_enabled()
        {
            self.poll_depth(session_data, false);
        }

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

    /// FD-mode upload: GPU-side copy from the imported dma-buf ring.
    /// Runs in `submit_impl` (GL context current). No CPU upload, no
    /// timestamp matching — the per-slot binary semaphore is the timeline.
    /// Quiet-path skip signal: warn once per reason episode so the next
    /// live run names the ack-freeze cause (stale vs dims vs acquire vs
    /// copy) instead of going silent. A changed reason re-warns;
    /// success clears it.
    fn note_upgrade_skip(&mut self, reason: &'static str) {
        if self.depth_upgrade_skip_reason != Some(reason) {
            self.depth_upgrade_skip_reason = Some(reason);
            warn!("depth upgrade skipped ({reason}); plain views");
        }
    }

    /// Loud-path (submit) skip signal (B1): warn once per reason episode,
    /// same pattern as [`note_upgrade_skip`], plus a heartbeat counter so
    /// SHM-read races attribute instead of silent-plain.
    fn note_submit_skip(&mut self, reason: &'static str) {
        self.depth_submit_skips += 1;
        if self.depth_submit_skip_reason != Some(reason) {
            self.depth_submit_skip_reason = Some(reason);
            warn!("depth submit skipped ({reason}); plain views");
        }
    }

    /// Ack-store failure signal (B4): count + warn once per episode.
    /// Success clears the episode so the next stall re-warns.
    fn note_ack_store_fail(&mut self, generation: u64) {
        self.depth_ack_store_fails += 1;
        if !self.depth_ack_store_warned {
            self.depth_ack_store_warned = true;
            warn!("depth ack store failed (gen {generation}); producer ring may stall");
        }
    }

    /// Acquire the depth image once per frame (B3). Returns the pending
    /// index, acquiring exactly once: a second call in the same frame
    /// (the end-frame upgrade re-copy) reuses the submit-time index with
    /// no second acquire. `None` on swapchain-level stall (warned by the
    /// caller with its path-appropriate skip reason).
    fn acquire_depth_image(&mut self) -> Option<usize> {
        if let Some(idx) = self.depth_pending_image {
            return Some(idx);
        }
        let idx: usize = {
            let data = self.depth_swapchain.as_mut()?;
            let idx = data.swapchain.acquire_image().ok()? as usize;
            // Bounded: never stall the app thread on a wedged swapchain.
            // A timed-out wait must release the image — otherwise each
            // event permanently consumes one swapchain image and depth
            // latches off instead of skipping a tick.
            if data
                .swapchain
                .wait_image(xr::Duration::from_nanos(5_000_000))
                .is_err()
            {
                let _ = data.swapchain.release_image();
                return None;
            }
            idx
        };
        self.depth_pending_image = Some(idx);
        #[cfg(test)]
        {
            self.depth_test_acquires += 1;
        }
        Some(idx)
    }

    /// Release the pending depth image exactly once (B3). No-op when none
    /// pending. Called at end of frame on every path and before any
    /// depth-swapchain drop.
    fn release_depth_pending(&mut self) {
        let pending = self.depth_pending_image.take();
        if pending.is_some() {
            #[cfg(test)]
            {
                self.depth_test_releases += 1;
            }
        }
        if let (Some(idx), Some(data)) = (pending, self.depth_swapchain.as_mut())
            && data.swapchain.release_image().is_err()
        {
            debug!("depth release_image failed (index {idx})");
        }
    }

    /// Shared depth poll for the submit path (`quiet=false`: loud,
    /// once-per-episode warns) and the end-frame upgrade (`quiet=true`:
    /// per-reason skip accounting via [`note_upgrade_skip`]).
    ///
    /// Callers own the render-state gates (`should_render`, null submits,
    /// submit-time already-armed) and the extension/master switches, which
    /// differ per path; this owns the shared tail: feed-open, the
    /// import-viability gate, control read, mode/generation filters, upload,
    /// and upgrade ok/pass counting. The quiet/loud reporting inside
    /// [`try_upload_depth_fd`] is untouched (R3/R4 depend on the
    /// distinction), as is every log line and counter update — the two call
    /// sites just share the boilerplate (S6).
    fn poll_depth(&mut self, session_data: &SessionData, quiet: bool)
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        if quiet {
            // Post-present upgrade: the Submit-time upload ran before
            // X-Plane's present for this frame, so it armed the previous
            // present's depth (systematic 1-frame lag). By end_frame the
            // present has completed; re-read the control header and upgrade
            // to the newest generation when the GL context allows. Quiet on
            // failure: the submit-time arming stays valid.
            if self.backend.depth_fd_import_viable() {
                if self.depth_feed.is_none() {
                    self.depth_feed = DepthFeed::open();
                }
                // Reason-coded read (B1): torn/recreate races attribute
                // via the upgrade skip reason instead of silent-plain.
                let upgrade: Option<DepthControl> = match self
                    .depth_feed
                    .as_ref()
                    .map(|f| f.read_control())
                {
                    None => None,
                    Some(Err(e)) => {
                        self.note_upgrade_skip(e.reason());
                        None
                    }
                    Some(Ok(ctl)) => Some(ctl),
                }
                .filter(|ctl| {
                    if ctl.mode != depth_feed::MODE_FD {
                        self.note_upgrade_skip("mode-not-fd");
                        return false;
                    }
                    if ctl.generation == 0 {
                        self.note_upgrade_skip("gen-zero");
                        return false;
                    }
                    if Some(ctl.generation) == self.depth_armed_generation {
                        return false;
                    }
                    true
                });
                if let Some(ctl) = upgrade {
                    let before = self.depth_armed_generation;
                    self.try_upload_depth_fd(session_data, &ctl, true);
                    if self.depth_armed_generation != before {
                        self.depth_upgrade_ok += 1;
                    } else {
                        self.depth_upgrade_pass += 1;
                    }
                }
            }
        } else {
            if self.depth_feed.is_none() {
                self.depth_feed = DepthFeed::open();
            }
            if self.depth_feed.is_none() {
                self.note_submit_skip("shm-absent");
                return;
            }
            // Backend viability first (no GL work without imports); the
            // upload itself re-validates mode, dims, FDs and staleness.
            if !self.backend.depth_fd_import_viable() {
                return;
            }
            // Reason-coded read (B1): absent/wrong-size/bad-magic/
            // bad-version/torn/stuck each warn once per episode plus a
            // heartbeat counter, instead of silent-plain.
            let ctl = match self
                .depth_feed
                .as_ref()
                .expect("feed opened above")
                .read_control()
            {
                Ok(ctl) => ctl,
                Err(e) => {
                    self.note_submit_skip(e.reason());
                    return;
                }
            };
            if ctl.mode != depth_feed::MODE_FD {
                self.note_submit_skip("mode-not-fd");
                return;
            }
            if ctl.generation == 0 {
                self.note_submit_skip("gen-zero");
                return;
            }
            self.try_upload_depth_fd(session_data, &ctl, false);
        }
    }

    /// The ack (`consumed_gen`) is stored only for fully-copied generations
    /// (fence-drained), so the producer never reuses an in-flight slot.
    /// Anything unexpected degrades to plain views for this frame.
    fn try_upload_depth_fd(&mut self, session_data: &SessionData, ctl: &DepthControl, quiet: bool)
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        if ctl.generation == 0 {
            return;
        }
        if !ctl.is_viable() {
            debug!(
                "depth not viable (mode {} fmt {} {}x{} epoch {} gen {}); plain views",
                ctl.mode, ctl.format, ctl.w, ctl.h, ctl.setup_epoch, ctl.generation,
            );
            return;
        }
        // Staleness against the header timestamp, sampled at use: the
        // end-frame upgrade reads generations published after frame start,
        // which a frame-start clock would misjudge as future stamps.
        let now = depth_feed::now_mono_ns();
        if ctl.is_stale(now, depth_feed::max_age_ms()) {
            if quiet {
                self.note_upgrade_skip("stale-shm");
            } else if !self.depth_stale_warned {
                self.depth_stale_warned = true;
                warn!("depth enabled but no fresh frame (SHM stale, dead, or producer stopped?)");
            }
            return;
        }
        self.depth_stale_warned = false;
        // Header-only clip: invalid falls back to plain. Error on the
        // submit path (once per episode via the stall flag), silent
        // debug on the quiet upgrade path.
        let (near_z, far_z) = match depth_feed::validate_near_far(ctl.near_z, ctl.far_z) {
            Some(v) => v,
            None => {
                if quiet {
                    self.note_upgrade_skip("near-far-invalid");
                } else if !self.depth_fail_warned {
                    self.depth_fail_warned = true;
                    error!(
                        "depth near/far invalid ({}, {}); plain views this frame",
                        ctl.near_z, ctl.far_z,
                    );
                }
                return;
            }
        };
        // Dims are authoritative per read: never stretch, skip the frame.
        let (cw, ch) = match self.swapchain_data.as_ref() {
            Some(c) => (c.info.width, c.info.height),
            None => return,
        };
        if cw == 0 || ctl.w != cw * 2 || ctl.h != ch {
            if quiet {
                self.note_upgrade_skip("dims-mismatch");
            } else {
                debug!(
                    "depth dims {}x{} vs color {cw}x{ch} — plain views this frame",
                    ctl.w, ctl.h,
                );
            }
            return;
        }
        let eye_w = ctl.eye_w();
        let eye_h = ctl.h;
        if !self.ensure_depth_swapchain(session_data, eye_w, eye_h, ctl.format) {
            if quiet {
                self.note_upgrade_skip("swapchain-unavailable");
            } else if !self.depth_fail_warned {
                self.depth_fail_warned = true;
                warn!(
                    "depth swapchain unavailable (generation {})",
                    ctl.generation
                );
            }
            return;
        }
        // B3: acquire once per frame; the upgrade re-copy reuses the same
        // index (acquire_depth_image returns the pending index with no
        // second acquire). No release here: end_frame releases exactly
        // once on every path, and drops release before any swapchain drop.
        let idx: usize = match self.acquire_depth_image() {
            Some(i) => i,
            None => {
                // Swapchain-level stall (wedged runtime or exhausted
                // images): silent here would look identical to a healthy
                // idle. Warned once per episode on the submit path, named
                // per reason on the quiet upgrade path.
                if quiet {
                    self.note_upgrade_skip("acquire-wait-timeout");
                } else if !self.depth_fail_warned {
                    self.depth_fail_warned = true;
                    warn!(
                        "depth swapchain acquire/wait timed out (generation {}); plain views this frame",
                        ctl.generation
                    );
                }
                return;
            }
        };
        let res = self.backend.import_and_copy_depth_fd(&DepthFdRequest {
            setup_epoch: ctl.setup_epoch,
            vk_format: ctl.format,
            mem_fds: ctl.mem_fds,
            sem_fds: ctl.sem_fds,
            generation: ctl.generation,
            depth_image_index: idx,
            eye_w,
            eye_h,
            alloc_bytes: ctl.alloc_bytes,
        });
        // B3 instrumentation: per-copy image index proves which released
        // image the runtime samples when the upgrade re-copies into the
        // same index.
        debug!(
            "depth copy gen {} -> image {idx} (epoch {}) copied={}",
            ctl.generation, ctl.setup_epoch, res.copied,
        );
        // Ack only fully-copied generations (fence-drained in the backend).
        // Skip the SHM write when the producer already observed it.
        // B4: count store failures in the heartbeat + throttled warn
        // (same once-per-episode style). No protocol change.
        if let Some(ack) = res.ack_gen
            && ack != ctl.consumed_gen
        {
            let stored = self
                .depth_feed
                .as_ref()
                .is_some_and(|feed| feed.store_consumed(ack));
            if stored {
                debug!("depth acked generation {ack} (was {})", ctl.consumed_gen);
                self.depth_ack_store_warned = false;
            } else {
                self.note_ack_store_fail(ack);
            }
        }
        if res.copied {
            // Header mapping range, sanitized for the runtime.
            let (min_depth, max_depth) = depth_feed::sanitize_min_max(ctl.min_depth, ctl.max_depth);
            self.depth_near_far = Some((near_z, far_z));
            self.depth_min_max = Some((min_depth, max_depth));
            self.depth_ready_this_frame = true;
            self.depth_armed_generation = Some(ctl.generation);
            self.depth_armed_time_ns = Some(ctl.time_ns);
            self.depth_fail_warned = false;
            self.depth_upgrade_skip_reason = None;
            self.depth_submit_skip_reason = None;
            if !self.depth_first_arm_logged {
                self.depth_first_arm_logged = true;
                info!(
                    "depth armed: {eye_w}x{eye_h}x2 fmt {} near {near_z} far {far_z}",
                    ctl.format,
                );
            }
        } else if quiet {
            // Quiet (end_frame upgrade) copy failures report via the
            // skip reason (once per reason episode), not silence: a
            // missing GL context there must not spam once per frame,
            // but the cause must be visible.
            self.note_upgrade_skip("copy-failed");
        } else if !self.depth_fail_warned {
            self.depth_fail_warned = true;
            warn!("depth FD copy failed (generation {})", ctl.generation);
        }
    }

    /// Ensure a per-eye depth swapchain for the mirror dims. Recreates on
    /// dims/format change. Creation failures back off 2 s and degrade to
    /// plain views — never stalls, never per-submit spam.
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
            && d.initial_format == want_fmt
        {
            return true;
        }
        // Past this point the old swapchain (if any) is dims-mismatched and
        // unusable: drop it so `Some` always means usable dims. Backend
        // import objects are epoch-keyed and re-created on demand.
        // B3: release the pending image BEFORE dropping the swapchain.
        self.release_depth_pending();
        self.depth_swapchain = None;
        if self.depth_swapchain_backoff.should_skip() {
            return false;
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
            .depth_swapchain_create_info(color_info, want_fmt)
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
                        self.depth_swapchain_backoff.note_failure();
                        return false;
                    }
                };
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
                self.depth_swapchain_backoff.note_success();
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
                                warn!("failed to enumerate depth swapchain images ({e})");
                                self.depth_swapchain_backoff.note_failure();
                                return false;
                            }
                        };
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
                        self.depth_swapchain_backoff.note_success();
                        true
                    }
                    Err(e2) => {
                        warn!("depth swapchain creation failed ({e2}); color-only");
                        self.depth_swapchain_backoff.note_failure();
                        false
                    }
                }
            }
        }
    }

    fn end_frame(
        &mut self,
        session_data: &SessionData,
        system: &System,
        display_time: xr::Time,
        overlays: Option<&OverlayMan>,
    ) -> xr::Result<()>
    where
        for<'b> &'b crate::overlay::AnySwapchainMap:
            TryInto<&'b crate::overlay::SwapchainMap<G::Api>, Error: std::fmt::Display>,
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq + Clone + std::fmt::Debug,
    {
        // Debug-only submit heartbeat: submits/s plus the depth-armed vs
        // plain split. Throttled 5 s; the submit path is untouched.
        // (S4: no separate total counter — every end takes exactly one of
        // the armed/plain branches, so the total is derived at emit time.)
        // B1+B4 additions: submit-skip counter (+reason) and ack-fail count
        // are surfaced here. Exact line change vs before:
        //   old: "... upgrade {} ok / {} pass, skip {}, repeat-ends {})"
        //   new: "... upgrade {} ok / {} pass, skip {}, submit-skip {} ({}), ack-fail {}, repeat-ends {})"
        {
            // Per-end stamp for the stamp-vs-render-pose audit: compare
            // against the waited-frame display_times in the same log.
            // A stamp newer than any wait serving this frame means the
            // submitted frame claims a time past its render poses.
            trace!(
                "end frame: display_time {:?} tid {:?}",
                display_time,
                std::thread::current().id()
            );
            // Repeat detection: same display_time as the previous end
            // means no new app frame (repeat re-present). Expected to
            // stay zero — repeats downstream happen in the runtime.
            if self.last_end_display_time == Some(display_time) {
                self.end_repeat_count += 1;
            }
            self.last_end_display_time = Some(display_time);
            let now = depth_feed::now_mono_ns();
            if self.submit_log_start_mono_ns == 0 {
                self.submit_log_start_mono_ns = now;
            } else if now.saturating_sub(self.submit_log_start_mono_ns) > 5_000_000_000 {
                let dt = (now - self.submit_log_start_mono_ns) as f64 / 1e9;
                let total = self.depth_armed_frames + self.depth_plain_frames;
                debug!(
                    "frames {:.1}/s ({} frames, {} with depth, {} plain, upgrade {} ok / {} pass, skip {}, submit-skip {} ({}), ack-fail {}, repeat-ends {})",
                    total as f64 / dt,
                    total,
                    self.depth_armed_frames,
                    self.depth_plain_frames,
                    self.depth_upgrade_ok,
                    self.depth_upgrade_pass,
                    self.depth_upgrade_skip_reason.unwrap_or("none"),
                    self.depth_submit_skips,
                    self.depth_submit_skip_reason.unwrap_or("none"),
                    self.depth_ack_store_fails,
                    self.end_repeat_count,
                );
                self.depth_armed_frames = 0;
                self.depth_plain_frames = 0;
                self.depth_upgrade_ok = 0;
                self.depth_upgrade_pass = 0;
                self.depth_submit_skips = 0;
                self.depth_ack_store_fails = 0;
                self.end_repeat_count = 0;
                self.submit_log_start_mono_ns = now;
            }
        }

        // B2: single end per frame. The builder below pre-validates
        // everything checkable (gates, upgrade, arming, clip, swapchains,
        // armed age) and returns whether to chain depth; the single
        // `stream.end` below then submits the selected layers exactly
        // once. Build-time failures yield a plain frame (same as today);
        // if the single end itself fails, the error returns with no
        // second end.
        // TEMPORARY rotation-only-group diagnostic (remove once pinned):
        // per-end depth identity. Trace volume is ~270 lines/s at 90 Hz
        // (2 per end + 1 per wait-frame), not ~18/s — keep trace-only.
        // A plain end means the server gets no
        // depth layer for this app frame and serves straight repeats
        // (client timewarp: rotation only) until depth resumes — the
        // prime suspect for the every-2nd/3rd-group symptom. Generation
        // must advance every end; a repeated gen is a reused (stale)
        // depth frame hiding inside a "with depth" count.
        let with_depth = self.depth_chain_ready(session_data);
        // B3: release the pending depth image exactly once, on every path
        // (depth-chained and all plain paths), before the single end.
        // The pending index was acquired at the submit-time copy; the
        // upgrade re-copied into the same index with no second acquire.
        self.release_depth_pending();
        trace!(
            "end frame: display_time {:?} tid {:?} depth {} gen {:?}",
            display_time,
            std::thread::current().id(),
            with_depth,
            self.depth_armed_generation,
        );
        if with_depth {
            self.depth_armed_frames += 1;
        } else {
            self.depth_plain_frames += 1;
        }

        // Layer build (shared for depth and plain): one view fetch, color
        // sub-images always; depth sub-images + DepthInfo chain only when
        // the builder armed depth. All checkable failures already returned
        // plain above, so this only fails closed on truly unexpected
        // shapes (fewer than 2 views → plain, same as today).
        let with_depth = with_depth && self.should_render
            && !self.submitting_null
            && self.eyes_submitted.iter().all(|eye| eye.is_some());
        let (near_z, far_z) = self.depth_near_far.unwrap_or((0.0, 0.0));
        let (min_depth, max_depth) = self.depth_min_max.unwrap_or((0.0, 1.0));
        let (eye_w, eye_h) = match self.depth_swapchain.as_ref() {
            Some(d) => (d.info.width, d.info.height),
            None => (0, 0),
        };

        let mut proj_layer_views = Vec::new();
        let mut depth_proj_views: Option<Vec<xr::CompositionLayerProjectionView<G::Api>>> = None;

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
            let eyes = self.eyes_submitted;
            if with_depth
                && let (Some(depth_data), true) = (
                    self.depth_swapchain.as_ref(),
                    eye_w != 0 && eye_h != 0,
                )
            {
                // Depth path: chained DepthInfo per eye.
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
                            .swapchain(&swapchain_data.swapchain)
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
                if color_subs.len() == 2 && depth_subs.len() == 2 {
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
                    let sys_views: Vec<xr::sys::CompositionLayerProjectionView> = (0..2)
                        .map(|i| xr::sys::CompositionLayerProjectionView {
                            ty: xr::sys::CompositionLayerProjectionView::TYPE,
                            next: &depth_infos[i] as *const _ as *const std::ffi::c_void,
                            pose: poses[i],
                            fov: fovs[i],
                            sub_image: *color_subs[i].as_raw(),
                        })
                        .collect();
                    // SAFETY: from_raw only wraps the structs; lifetimes
                    // are preserved because swapchains, views, and
                    // depth_infos all outlive the synchronous end() call.
                    let proj_views: Vec<xr::CompositionLayerProjectionView<G::Api>> =
                        sys_views
                            .into_iter()
                            .map(|s| unsafe {
                                xr::CompositionLayerProjectionView::from_raw(s)
                            })
                            .collect();
                    trace!("projection layer present (with depth)");
                    depth_proj_views = Some(proj_views);
                }
            }
            if depth_proj_views.is_none() {
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
                        } = eyes[eye_index]
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
                    .collect();
                if !with_depth {
                    trace!("projection layer present");
                }
            }
        }

        // Single end with the selected layers (B2): exactly one
        // `stream.end` per frame (one textual call site below). A
        // depth-end failure returns the error with no second (plain) end;
        // build-time failures already selected the plain set above.
        // Note: depth_proj_views borrows self (swapchains) while stream
        // needs &mut self.stream — disjoint fields, so split the borrows.
        let depth_selected = depth_proj_views.is_some();
        let plain_proj_layer = if depth_selected || proj_layer_views.is_empty() {
            None
        } else {
            trace!("projection layer present");
            Some(
                xr::CompositionLayerProjection::new()
                    .space(session_data.tracking_space())
                    .views(&proj_layer_views),
            )
        };
        let depth_proj_layer = depth_proj_views.as_ref().map(|depth_views| {
            trace!("projection layer present (with depth)");
            xr::CompositionLayerProjection::new()
                .space(session_data.tracking_space())
                .views(depth_views)
        });
        let proj_layer_ref = depth_proj_layer
            .as_ref()
            .or(plain_proj_layer.as_ref());
        let mut layers: Vec<&xr::CompositionLayerBase<_>> = Vec::new();
        if let Some(l) = proj_layer_ref {
            layers.push(l);
        }
        let overlay_layers;
        if let Some(overlay_man) = overlays {
            overlay_layers = overlay_man.get_layers(session_data, self.app_fade_grid);
            layers.extend(overlay_layers.iter().map(Deref::deref));
        }
        let end_result = self
            .stream
            .end(display_time, xr::EnvironmentBlendMode::OPAQUE, &layers);
        if end_result.is_ok() && depth_selected {
            trace!("frame submitted (with depth)");
            self.depth_fail_warned = false;
        }
        match end_result {
            Ok(()) => {
                trace!("frame submitted");
                Ok(())
            }
            Err(e) => {
                // No second end (B2): surface the single-end failure.
                // Throttle like other depth failures when depth was
                // selected; plain-end failures stay loud (as today via
                // unwrap — now an explicit error return instead).
                if depth_proj_views.is_some() && !self.depth_fail_warned {
                    self.depth_fail_warned = true;
                    warn!("depth frame end failed ({e}); no plain retry (single end)");
                }
                Err(e)
            }
        }
    }

    /// Layer builder (B2) + chain-time gates (B7): decides whether the
    /// single `stream.end` in [`end_frame`] chains depth. Performs the
    /// post-present upgrade, then pre-validates everything checkable at
    /// build time (extension, master, arming, clip, swapchain dims, armed
    /// age) so the depth end is only attempted when likely to succeed.
    /// Returns true when depth should be chained, false for plain (same
    /// app-visible plain-frame behavior as today for every build-time
    /// failure). Performs no `stream.end` itself.
    fn depth_chain_ready(&mut self, session_data: &SessionData) -> bool
    where
        for<'a> &'a openxr_data::GraphicalSession:
            TryInto<&'a openxr_data::Session<G::Api>, Error: std::fmt::Display>,
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
        // Post-present upgrade (shared `poll_depth` tail, quiet accounting):
        // re-read the control header and upgrade to the newest generation
        // when the GL context allows. Quiet on failure: the submit-time
        // arming stays valid.
        self.poll_depth(session_data, true);
        // Armed by try_upload_depth_fd during Submit (or upgraded above).
        // Both halves were handled atomically there: both eyes or neither.
        if !self.depth_ready_this_frame {
            return false;
        }
        if self.depth_near_far.is_none() {
            self.note_upgrade_skip("no-near-far");
            return false;
        }
        // B7: re-validate the armed depth age at chain time. The submit
        // copy checked staleness then; a 100ms-boundary flap between
        // submit and end must degrade to plain with a reason.
        let armed_time = self.depth_armed_time_ns.unwrap_or(0);
        let now = depth_feed::now_mono_ns();
        if depth_feed::depth_armed_is_stale(armed_time, now, depth_feed::max_age_ms()) {
            self.note_upgrade_skip("armed-stale");
            return false;
        }
        let (eye_w, eye_h) = match self.depth_swapchain.as_ref() {
            Some(d) => (d.info.width, d.info.height),
            None => {
                self.note_upgrade_skip("no-depth-swapchain");
                return false;
            }
        };
        if eye_w == 0 || eye_h == 0 {
            self.note_upgrade_skip("bad-depth-dims");
            return false;
        }
        if self.swapchain_data.is_none() {
            self.note_upgrade_skip("no-color-swapchain");
            return false;
        }
        true
    }
}

// Note: no Drop impl for FrameController on purpose. A Drop would forbid
// moving `backend`/`swapchain_data` out of the controller (session-restart
// harvest in `post_session_restart`), so every live path releases the
// pending depth image explicitly instead (end of frame on every path;
// before any depth-swapchain drop; old-controller harvest).

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
        /// B3 test hook: import-viable toggle (no GL). Default false
        /// (inert, as before); tests opt into viable copies.
        pub fake_depth_viable: bool,
        /// B3 test hook: recorded (generation, image index) per copy, no GL.
        /// Proves one acquire per frame (submit+upgrade share the index).
        pub fake_depth_copies: Vec<(u64, usize)>,
        depth_images_len: usize,
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

        /// Test-only depth posture: capable enough to open the feed and read
        /// the control header, but never import (no GL context in tests).
        /// Lets Tier-2 prove the gate chain runs without arming anything.
        fn supports_depth() -> bool {
            true
        }

        fn depth_format_for_header(
            vk_format: i32,
        ) -> Option<<Self::Api as openxr::Graphics>::Format> {
            // Fake maps both SHM formats to the fakexr-supported format 0
            // so ensure_depth_swapchain can create a real (fake) swapchain
            // with no GL. Unknown formats stay unsupported.
            match vk_format {
                crate::depth_feed::FORMAT_D32F | crate::depth_feed::FORMAT_D32S8 => Option::Some(0),
                _ => Option::None,
            }
        }

        fn depth_fd_import_viable(&self) -> bool {
            self.fake_depth_viable
        }

        fn depth_swapchain_create_info(
            &self,
            color_info: &xr::SwapchainCreateInfo<Self::Api>,
            depth_format: <Self::Api as xr::Graphics>::Format,
        ) -> Option<xr::SwapchainCreateInfo<Self::Api>> {
            if color_info.width == 0 || color_info.height == 0 {
                return Option::None;
            }
            Some(xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::DEPTH_STENCIL_ATTACHMENT,
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
            _format: <Self::Api as xr::Graphics>::Format,
        ) {
            // fakexr enumerates 0 images; record the length but accept any
            // index in the copy (the index comes from the fake swapchain
            // acquire, expected 0).
            self.depth_images_len = images.len();
        }

        fn import_and_copy_depth_fd(
            &mut self,
            req: &crate::graphics_backends::DepthFdRequest,
        ) -> crate::graphics_backends::DepthFdResult {
            use crate::graphics_backends::DepthFdResult;
            if !self.fake_depth_viable {
                return DepthFdResult {
                    copied: false,
                    ack_gen: Option::None,
                };
            }
            if req.generation == 0 || req.eye_w == 0 || req.eye_h == 0 {
                return DepthFdResult {
                    copied: false,
                    ack_gen: Option::None,
                };
            }
            if req.mem_fds.iter().any(|&f| f <= 2) || req.sem_fds.iter().any(|&f| f <= 2) {
                return DepthFdResult {
                    copied: false,
                    ack_gen: Option::None,
                };
            }
            // No GL: record the copy and report success with an ack.
            // The image index must be the pending (single-acquire) index;
            // the test asserts submit+upgrade share it.
            self.fake_depth_copies.push((req.generation, req.depth_image_index));
            DepthFdResult {
                copied: true,
                ack_gen: Some(req.generation),
            }
        }

        fn drop_depth_imports(&mut self) {
            self.depth_images_len = 0;
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
                fake_depth_viable: false,
                fake_depth_copies: Vec::new(),
                depth_images_len: 0,
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
        // Viable v2 FD segment at the fixed path + master on + extension on,
        // but the Fake backend reports no import viability (no GL context):
        // the submit path opens the feed yet arms nothing and acks nothing.
        // (Feed reads stay harness-side: without GL the controller never
        // reaches read_control, so this is an inertness + feed-lifecycle
        // tripwire, not gate-chain coverage — logic is Tier-1 covered.)
        let _lock = crate::depth_feed::env_guard();
        let seg_name = crate::depth_feed::DEFAULT_SHM_NAME;
        // Shared SHM test helper (S7): same viable segment as the hand-rolled
        // 128B block this replaced (20x10, gen 1, epoch 3, fresh stamp).
        let _seg = crate::depth_feed::make_seg_v2(
            seg_name,
            20,
            10,
            1,
            crate::depth_feed::MODE_FD,
            |h| {
                h.setup_epoch = 3;
                h.alloc_bytes = 800;
            },
        );
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");

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
            ctrl.set_depth_extension_enabled(true);
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
            assert!(
                !ctrl.depth_ready_this_frame,
                "import-viable gate must block arming (Fake has no GL context)"
            );
            assert!(
                ctrl.depth_near_far.is_none(),
                "import-viable gate must block arming (Fake has no GL context)"
            );
            assert!(
                ctrl.depth_min_max.is_none(),
                "import-viable gate must block arming (Fake has no GL context)"
            );
            assert!(
                ctrl.depth_armed_generation.is_none(),
                "import-viable gate must block arming (Fake has no GL context)"
            );
            // The gate chain ran (feed opened) but nothing armed…
            assert!(ctrl.depth_feed.is_some(), "feed must open past master gate");
        }
        // …and the published header still reads viable with no ack written.
        {
            let feed = crate::depth_feed::DepthFeed::open().expect("feed opens");
            let ctl = feed.read_control().expect("control readable");
            assert!(ctl.is_viable());
            assert_eq!(ctl.generation, 1);
            assert_eq!(ctl.consumed_gen, 0);
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
        // (`_seg`'s Drop unlinks the fixed path here.)
    }

    #[test]
    fn depth_inert_with_extension_disabled() {
        // Master switch on and a depth-capable backend, but the runtime did
        // not advertise khr_composition_layer_depth: the extension gate
        // short-circuits before the feed is even opened — plain views,
        // no depth state at all. Also pins the wired (non-)mapping of an
        // unknown Vulkan format: no depth swapchain, still plain.
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        // Defensive: the fixed path must be absent for this test.
        unsafe {
            let cname = std::ffi::CString::new(crate::depth_feed::DEFAULT_SHM_NAME).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }

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
            assert!(
                ctrl.depth_swapchain.is_none(),
                "extension gate must block depth before any swapchain exists"
            );
            assert!(
                ctrl.depth_feed.is_none(),
                "extension gate must short-circuit before the feed opens"
            );
            assert!(
                !ctrl.depth_ready_this_frame,
                "extension gate must block arming (khr_composition_layer_depth off)"
            );
            // Extension probe propagated (false here): the gate never opens.
            assert!(
                !ctrl.depth_extension_enabled,
                "extension gate must stay closed (probe false in this test)"
            );
        }
        // Unknown Vulkan formats map to no GL depth format: the swapchain
        // is never created, frames stay plain.
        {
            let data = f.comp.openxr.session_data.get();
            let mut frame_lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = frame_lock.as_mut().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(
                !ctrl.ensure_depth_swapchain(&data, 10, 10, 96),
                "format gate must reject unknown Vulkan format 96"
            );
            assert!(
                ctrl.depth_swapchain.is_none(),
                "format gate must leave no depth swapchain for unknown format 96"
            );
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }

    #[test]
    fn depth_submit_skip_reason_coded_absent() {
        // B1: submit-time SHM absence attributes via warn-once-per-episode
        // + heartbeat counter instead of silent-plain.
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        unsafe {
            let cname = std::ffi::CString::new(crate::depth_feed::DEFAULT_SHM_NAME).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }

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
            ctrl.set_depth_extension_enabled(true);
            ctrl.backend.fake_depth_viable = true;
        }
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert_eq!(
                ctrl.depth_submit_skip_reason,
                Some("shm-absent"),
                "absent SHM must attribute as shm-absent"
            );
            assert!(
                ctrl.depth_submit_skips >= 1,
                "submit-skip counter must count the miss"
            );
            assert!(
                !ctrl.depth_ready_this_frame,
                "absent SHM must not arm"
            );
        }
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }

    #[test]
    fn depth_single_acquire_release_per_frame() {
        // B3: one depth acquire per frame across submit+upgrade, released
        // exactly once; upgrade re-copies into the SAME index.
        let _lock = crate::depth_feed::env_guard();
        let seg_name = crate::depth_feed::DEFAULT_SHM_NAME;
        let _seg = crate::depth_feed::make_seg_v2(
            seg_name,
            20,
            10,
            1,
            crate::depth_feed::MODE_FD,
            |h| {
                h.setup_epoch = 3;
                h.alloc_bytes = 800;
            },
        );
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");

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
            ctrl.set_depth_extension_enabled(true);
            ctrl.backend.fake_depth_viable = true;
        }
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        // Publish a newer generation before end: the upgrade must re-copy
        // into the same index with no second acquire.
        drop(_seg);
        let _seg2 = crate::depth_feed::make_seg_v2(
            seg_name,
            20,
            10,
            2,
            crate::depth_feed::MODE_FD,
            |h| {
                h.setup_epoch = 3;
                h.alloc_bytes = 800;
            },
        );
        f.comp.PostPresentHandoff();
        f.check_frame_state(fakexr::FrameState::Ended);

        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert_eq!(
                ctrl.depth_armed_generation,
                Some(2),
                "upgrade must re-arm the newer generation"
            );
            assert_eq!(
                ctrl.depth_test_acquires, 1,
                "exactly one depth acquire per frame across submit+upgrade"
            );
            assert_eq!(
                ctrl.depth_test_releases, 1,
                "exactly one depth release per frame (acquire==release)"
            );
            assert!(
                ctrl.backend.fake_depth_copies.len() >= 2,
                "submit copy + upgrade re-copy expected, got {:?}",
                ctrl.backend.fake_depth_copies
            );
            let first_idx = ctrl.backend.fake_depth_copies[0].1;
            for (_, idx) in &ctrl.backend.fake_depth_copies {
                assert_eq!(
                    *idx, first_idx,
                    "upgrade must re-copy into the SAME index (no second acquire)"
                );
            }
            assert!(
                ctrl.depth_pending_image.is_none(),
                "pending image must be released at end of frame"
            );
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }

    #[test]
    fn depth_armed_stale_plain_at_chain() {
        // B7: armed depth re-validated at chain time; an aged arming goes
        // plain with a reason instead of chaining stale depth.
        let _lock = crate::depth_feed::env_guard();
        let seg_name = crate::depth_feed::DEFAULT_SHM_NAME;
        let _seg = crate::depth_feed::make_seg_v2(
            seg_name,
            20,
            10,
            5,
            crate::depth_feed::MODE_FD,
            |h| {
                h.setup_epoch = 3;
                h.alloc_bytes = 800;
            },
        );
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");

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
            ctrl.set_depth_extension_enabled(true);
            ctrl.backend.fake_depth_viable = true;
        }
        assert_eq!(f.submit(vr::EVREye::Left), None);
        assert_eq!(f.submit(vr::EVREye::Right), None);
        // Age the arming past the 100ms bound before end.
        {
            let data = f.comp.openxr.session_data.get();
            let mut frame_lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = frame_lock.as_mut().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert!(ctrl.depth_ready_this_frame, "submit must arm fresh depth");
            let now = crate::depth_feed::now_mono_ns();
            ctrl.depth_armed_time_ns = Some(now.saturating_sub(500_000_000));
        }
        f.comp.PostPresentHandoff();
        // Single end, plain: Ended state proves exactly one end call (B2
        // pattern); the armed-stale reason proves chain-time revalidation.
        f.check_frame_state(fakexr::FrameState::Ended);
        {
            let data = f.comp.openxr.session_data.get();
            let lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = lock.as_ref().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert_eq!(
                ctrl.depth_upgrade_skip_reason,
                Some("armed-stale"),
                "aged arming must attribute armed-stale"
            );
            assert!(
                ctrl.depth_pending_image.is_none(),
                "plain path must still release the pending image"
            );
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }

    #[test]
    fn depth_ack_store_fail_counted() {
        // B4: ack-store failures count + warn once per episode.
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        unsafe {
            let cname = std::ffi::CString::new(crate::depth_feed::DEFAULT_SHM_NAME).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }

        let f = Fixture::new();
        f.ensure_real_session(false);
        {
            let data = f.comp.openxr.session_data.get();
            let mut frame_lock = data.comp_data.0.lock().unwrap();
            let DynFrameController::Fake(ctrl) = frame_lock.as_mut().unwrap() else {
                panic!("Frame controller was not set up or not faked!");
            };
            assert_eq!(ctrl.depth_ack_store_fails, 0);
            ctrl.note_ack_store_fail(7);
            assert_eq!(ctrl.depth_ack_store_fails, 1);
            assert!(ctrl.depth_ack_store_warned);
            ctrl.note_ack_store_fail(8);
            assert_eq!(ctrl.depth_ack_store_fails, 2, "counter counts every miss");
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }

    #[test]
    fn depth_plain_end_single_end_reaches_ended() {
        // B2: restructured single-end plain path reaches Ended exactly
        // once (mirrors the existing fakexr FrameState::Ended pattern).
        let _lock = crate::depth_feed::env_guard();
        crate::depth_feed::set_env_for_tests("XRIZER_DEPTH_SHM", "1");
        unsafe {
            let cname = std::ffi::CString::new(crate::depth_feed::DEFAULT_SHM_NAME).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }

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
            assert!(
                ctrl.depth_plain_frames >= 1,
                "plain build failure must count a plain end"
            );
            assert_eq!(ctrl.depth_armed_frames, 0);
        }

        crate::depth_feed::del_env_for_tests("XRIZER_DEPTH_SHM");
    }
}
