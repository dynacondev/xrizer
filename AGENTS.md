# AGENTS.md

Rust cdylib: reimplements OpenVR on top of OpenXR. Entry points are `VRClientCoreFactory` / `HmdSystemFactory` in `src/lib.rs`; one module per OpenVR interface (`src/system.rs`, `src/compositor.rs`, `src/input/`, `src/overlay.rs`, …).

## Build — use `cargo xbuild`, not `cargo build`

- Dev: `cargo xbuild`
- Release: `cargo xbuild --release`
- CI variant: `cargo xbuild --release --features static-openxr --verbose`

`xbuild` is a cargo alias (`run --package xbuild --`, see `.cargo/config.toml`). It builds `-p xrizer`, then symlinks the cdylib as `bin/linux64/vrclient.so` (name/dir from `build.rs` envs `XRIZER_OPENVR_PLATFORM_DIR` / `XRIZER_OPENVR_VRCLIENT_NAME`), writes `openvrpaths.vrpath`, and touches `bin/version.txt`. `target/{debug,release}/` is then directly usable as an OpenVR runtime dir. Plain `cargo build` skips all of that.

## Verify

Run in this order (mirrors CI in `.github/workflows/ci.yml`):

- `cargo test --verbose`
- `cargo +nightly miri test` (flags preset via `MIRIFLAGS` in `.cargo/config.toml`)
- `cargo fmt --check`
- `cargo clippy --workspace --all-targets` (short alias: `cargo c`)

Notes:

- `clippy::all` is `deny` (`[workspace.lints.clippy]` + `#![deny(clippy::all)]` in `src/lib.rs`). Keep it clean.
- `--workspace` covers only `xrizer` + `xbuild` (`[workspace] members = ["xbuild"]`). `openvr/`, `fakexr/`, `macros/`, `shaders/` are path dependencies, not workspace members.
- `tests/tests.rs::smoke_test` builds the cdylib via `test-cdylib` + loads it with `libloading`; it is `#[cfg_attr(miri, ignore)]`.
- Most unit tests live in `src/input/legacy.rs`, `src/input/custom_bindings.rs`, `src/rendermodels.rs` and drive the mock OpenXR runtime in `fakexr/` (dev-dependency, "easily manipulatable OpenXR runtime for testing"). Prefer extending `fakexr` helpers over real hardware/OpenXR in tests.

## Codegen — don't hand-edit generated code

- `openvr/` build script generates Rust bindings from `openvr/headers/` via bindgen + trait-unification parsing (see `openvr/README.md`). Fix the generator/parsing, not the output.
- Root `build.rs` compiles shaders via the `shaders/` crate (`shaders::compile`). It also computes the reported version: `XRIZER_VERSION` env wins if set, else `git describe` via `vergen-gitcl` (`git-version` default feature), else `CARGO_PKG_VERSION`. Tarball/no-git builds need `--no-default-features --features monado` (or similar) + `XRIZER_VERSION=…`.
- Default features: `monado`, `git-version`. Others: `static-openxr` (CI release), `tracing`.

## Env vars / debugging

- `RUST_LOG` controls logging; extra targets: `openvr_calls` (each OpenVR fn), `tracked_property`.
- `XRIZER_CUSTOM_BINDINGS_DIR`, `XRIZER_TRACKER_SERIALS` (`;`-separated serials for FBT trackers), `XRIZER_INSTALL_PREFIX` (xbuild `openvrpaths.vrpath.in` substitution), `VR_OVERRIDE=/path/to/xrizer` (game launch override).
- Runtime log for bug reports: `$XDG_STATE_HOME/xrizer/xrizer.txt` (else `~/.local/state/xrizer/xrizer.txt`).

## Toolchain / system deps

- Requires Rust ≥1.85 (edition 2024); `mise.toml` pins a newer toolchain. Linux CI builds in Ubuntu 22.04 container with `libvulkan-dev libopenxr-dev libx11-xcb-dev shaderc/glslc` (`.github/workflows/ubuntu.dockerfile`).
