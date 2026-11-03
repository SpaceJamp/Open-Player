# Player UI: on-screen controls with Dear ImGui

Date: 2026-11-03. Status: approved in chat, awaiting review of this written spec.

## Goal

Make the player window usable without the keyboard or the launcher, the way PotPlayer is: on-screen controls,
opening files from the window, visible VSR feedback, and a right-click menu. One video at a time.

**Success:** normal use (play, pause, seek, volume, VSR on/off, fullscreen, opening another file) needs only the
mouse; the overlay costs at most about 1 ms per frame and adds no dropped frames at 1080p60 -> 4K.

What the user asked for: on-screen controls, opening files (Open dialog, drag and drop, recent files), VSR
feedback, a right-click menu; drawn with Dear ImGui. Assumptions they approved: controls auto-hide, the info
overlay toggles with a key, opening another file restarts the player.

## Out of scope

- Playlists, next/previous video.
- Audio-track switching (the player opens one audio track; that is a separate change in `Media`).
- Choosing a monitor from the menu (window placement doesn't work on WSL; `--screen` stays).
- Settings dialogs, skins, subtitles.

## Dependency

Dear ImGui **v1.93.0** (MIT), vendored in `player/third_party/imgui/`: `imgui.cpp`, `imgui_draw.cpp`,
`imgui_tables.cpp`, `imgui_widgets.cpp`, their headers (`imgui.h`, `imgui_internal.h`, `imconfig.h`,
`imstb_*.h`), `LICENSE.txt`, and the backends `backends/imgui_impl_sdl3.{h,cpp}` and
`backends/imgui_impl_vulkan.{h,cpp}`. Not `imgui_demo.cpp`. Built as part of the `open-player` target in
`player/CMakeLists.txt`; no system package.

## 1. Drawing (`player/display.{h,cpp}`)

Today `Display::present(w, h, stream)` copies the RGBA frame into an image, then blits (scales) it onto the
swapchain image, which goes straight to `PRESENT_SRC`.

New order inside `present`:

1. If the video frame changed since the last present: copy buffer -> image as today (skipped when only the UI
   changed, so UI redraws don't pay the WSL frame copy). A `bool new_frame` argument says which.
2. Clear and blit to the swapchain image as today.
3. Transition the swapchain image to `COLOR_ATTACHMENT_OPTIMAL`, begin a render pass that loads (keeps) the
   blitted pixels, record ImGui's draw data (`ImGui_ImplVulkan_RenderDrawData`), end the pass; the pass's final
   layout is `PRESENT_SRC_KHR`.

Needed for that:

- Swapchain images also get `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` (checked against the surface's supported usage).
- One `VkRenderPass` (color attachment, `LOAD_OP_LOAD`, `STORE_OP_STORE`, initial layout
  `COLOR_ATTACHMENT_OPTIMAL`, final layout `PRESENT_SRC_KHR`), plus an image view and framebuffer per swapchain
  image, recreated with the swapchain.
- A small descriptor pool for ImGui's font texture.
- A classic render pass, not dynamic rendering: WSL's Vulkan driver (dzn) may not offer Vulkan 1.3 or
  `VK_KHR_dynamic_rendering`.
- `Display` exposes what `ImGui_ImplVulkan_Init` needs (instance, physical device, device, queue family and
  queue, render pass, image count) through one `init_imgui()` method, so ImGui setup stays in the display layer.

The CUDA seek bar drawn into the video (`progress` argument of `rgb_to_rgba8` in `player/kernels.{h,cu}`) is
removed: ImGui draws the real one.

## 2. UI (`player/ui.{h,cpp}`, new; event wiring in `player/main.cpp`)

The UI code lives in its own file with a narrow interface: each frame, `main.cpp` passes a small state struct
(position, duration, paused, volume, VSR available/on, fullscreen, info-overlay text, recent files) and gets
back a list of actions (toggle pause, seek to t, set volume, toggle VSR, toggle fullscreen, open path, quit).
`main.cpp` applies those through the code paths the keyboard shortcuts already use, so a click and a key do the
same thing.

- **Control bar** (bottom, full width): play/pause button, `12:34 / 45:00`, seek slider (seeks on release, shows
  the target time while dragging), volume slider, VSR toggle, fullscreen button, Open button. Shown on mouse
  movement, hidden after 2 s without movement, always shown while paused or while the mouse is over the bar.
- **Info overlay** (top-left, toggled with `I`, off by default): VSR on/off; from -> to resolution (for example
  `1920x1080 -> 3840x2160`, or `1920x1080 shrunk to 1280x720 -> 2560x1440`); decode path (`NVDEC h264 nv12`);
  fps vs the video's fps and dropped frames, from the same per-second numbers `--stats` reports.
- **Right-click menu:** Open..., Recent files (submenu), VSR on/off, Info overlay, Fullscreen, Quit.
- **Opening a file:** Open uses `SDL_ShowOpenFileDialog` (asynchronous, result arrives via a callback);
  dropping a file on the window arrives as `SDL_EVENT_DROP_FILE`. Either way the path is added to recent files
  and the player re-executes itself (`execv` of its own path) with the same arguments and the new video, so the
  upscaler plan and engine are rebuilt for the new resolution.
- **Recent files:** last 10 paths, newest first, no duplicates, missing files dropped when shown; stored one
  per line in `$XDG_CONFIG_HOME/open-player/recent.txt` (default `~/.config/open-player/recent.txt`).
- **Window icon:** `assets/icon-64.png`, found relative to the executable (`SDL_GetBasePath()/../assets/`),
  loaded with SDL3's `SDL_LoadPNG` and set with `SDL_SetWindowIcon`; a missing file is ignored.
- **Window title:** the file name only; the position, VSR and volume info move to the control bar and overlay.
- **Keyboard:** every existing shortcut stays; `I` is new. Keys go to ImGui only while a menu or dialog has focus.

## Redraws and speed

The player redraws only when something changed (`dirty`), because each redraw used to copy a full frame. With
the UI, a redraw also happens on mouse movement, while the control bar fades or a slider is dragged, and when
the bar hides. Those redraws pass `new_frame = false`, so they skip the frame copy.

## Errors

- ImGui or the render pass failing to initialise is reported and the player still plays without the UI.
- A failed `execv` reports the error and keeps playing the current video.
- An unreadable recent-files file is treated as empty; failures to write it are ignored.

## Testing

- `tests/smoke.sh` and `tests/codecs.sh` keep running offscreen with the UI on, so a crash, a stuck frame or a
  Vulkan error in the new drawing code fails them.
- Speed: `--stats` at 1080p60 -> 4K with the fast model, before and after; the overlay must cost at most about
  1 ms per frame and add no dropped frames.
- Recent-files logic (cap 10, no duplicates, newest first, missing files dropped) is a pure function, checked
  by a new `open-player --uitest` flag that needs no window or GPU; `tests/smoke.sh` runs it.
- By hand, by the user: seek dragging, the right-click menu, the open dialog, drag and drop, auto-hide.
