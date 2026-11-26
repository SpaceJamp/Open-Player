# Captions (English subtitles)

Date: 2026-11-25. Status: draft, waiting for the user's approval. Reviewed by a builder check; its questions are
answered below as my calls.

## Intent

The user wants closed captions in Open Player, mostly in English. Today the player opens one video and one audio
stream and ignores every subtitle stream, although the user's videos carry them: most of their WEB-DL files have
20-40 SubRip (text) tracks including 1-4 English ones (plain, "British", "SDH"/hearing-impaired, forced), and
Blu-ray rips carry PGS (picture) subtitles (e.g. The Flintstones: one English SDH PGS track). None of their files
has EIA-608 captions inside the video stream, and none has a separate .srt file.

## Goals

- Show English captions over the video, sharp at any size (drawn after upscaling, never through RTX VSR or Pixel
  Flow), for both text (SubRip, plus ASS/SSA/mov_text/WebVTT as plain text) and picture (PGS, DVD/VobSub) tracks.
- Turn them on and off quickly, choose between the English tracks, and have the choice remembered.
- `.srt` files next to the video count as tracks too (`movie.srt`, `movie.en.srt`, `movie.eng.srt`).

## Non-goals (for now)

- Other languages: the track list offers English tracks only (language `eng`/`en`, plus untagged tracks and
  `.srt` files without a language). Non-Latin scripts (CJK, Arabic, Hebrew, Thai) would need new fonts and shaping.
- ASS styling (colours, positions, karaoke), libass, burning captions into the video.
- EIA-608/708 captions inside the video stream; downloading subtitles; a timing-offset control.
- Linux: keep it building; test on Windows only (standing rule).

## Anti-goals

- Captions that are blurry or scaled with the video's pixels, or that the upscaler touches.
- Captions out of step with the picture after a seek or a pause.
- Slowing playback down: subtitle work never blocks the video or audio path.

## Approaches

- **A. Drawn by the player's own UI layer (chosen by the user).** FFmpeg decodes the chosen track; text is drawn
  with the Inter font through Dear ImGui after the frame is composed; picture subtitles are drawn as textures. No new
  dependencies. Plain text only (styling dropped).
- **B. libass via FFmpeg's subtitles filter.** Full styling and all scripts, but a large new dependency chain
  (libass, HarfBuzz, FreeType, fontconfig) and captions burnt in before upscaling (smeared) unless composited
  separately. Dropped.
- **C. Like A, text tracks only.** Simpler, but Blu-ray rips (PGS) would have no captions. Dropped.

## Design (decided)

### Choosing a track (my calls)

- A file's caption tracks: its English subtitle streams in file order, then matching `.srt` files beside it.
  Each is named from its title tag, else "English", plus "(SDH)" when the title says SDH or the stream has the
  hearing-impaired flag, "(forced)" for the forced flag.
- Default track when captions are on: the first SDH one (closed captions describe sounds too), else the first
  English one that isn't forced-only, else the first English one.
- **Captions on/off is remembered** in settings.txt (`captions=1`), off by default: C, the CC button and the menu
  all save it, like the volume (my call: a captions viewer wants them on in every film without a trip to Settings).
  The chosen track is not remembered per file: each file starts on its default track.
- **Forced English tracks show even with captions off** (my call; what other players do): they carry the
  translation of foreign-language lines that belong to the film. With captions on, the chosen track is shown
  instead (a full track includes those lines).

### Controls

- **C** key toggles captions (a notice: "Captions: English (SDH)" / "Captions off"; "No English captions" when the
  file has none).
- A **CC** button in the control bar between the volume and the upscaling badge, highlighted when on (icon
  `ICON_BADGE_CC` from Tabler, text "CC" without the icon font). Hidden when the file has no English tracks.
- The right-click menu gets a **Captions** submenu: Off, then each English track (the current one ticked).
- Settings > General: a **Captions** switch, the same remembered on/off as C.

### Timing and decoding

- `Media` opens the chosen subtitle stream's decoder alongside video and audio; the demux loop already sees every
  packet and hands the chosen stream's packets to `avcodec_decode_subtitle2`. Each result becomes a cue
  `{start, end, text}` or `{start, end, RGBA image, its rectangle in video coordinates}` in a small locked list.
- Switching tracks or turning captions on mid-file: re-seek to the current position (as the decoder change does),
  so cues around now are read again.
- After a seek the cue list is cleared; cues come back as the demuxer reaches them. A line that began before the
  seek point and is still on screen may be missed (shortcut: Matroska stores no "active at" index; acceptable).
- `.srt` beside the video: opened with its own small FFmpeg context and read whole at open (a few hundred KB).
  Not UTF-8 (an old Windows-1252 file): read as Windows-1252 instead of showing garbage (my call).
- Drawing picks the cues whose `start <= pos < end` (pos = the shown frame's time, so pause and seek just work);
  a cue without an end lasts until the next one or 5 s.

### Drawing

- Text: bottom centre, white Inter text with a black outline (8-direction offset) for any background, size 4.6%
  of the window's height (about 50 px at 1080p, 100 px at 4K), wrapped at 80% of the width, at most 3 lines.
  HTML-ish tags in SubRip (`<i>`, `<b>`, `<font>`) and ASS override blocks (`{\...}`) are stripped; `\N` is a line
  break.
- Lifted above the control bar while it's visible, so they never overlap.
- Picture subtitles: converted to RGBA on the decode thread, uploaded as an ImGui (Vulkan) texture on the main
  thread, and drawn at their own position, mapped from the subtitle's canvas (the decoder's width x height, e.g.
  1920x1080) **centred on the shown picture** (a 1920x800 cropped Blu-ray picture sits in the middle of that canvas),
  then **clamped into the window** and lifted above the control bar like text (my call: never lost in the bars or
  off-screen). Old textures are freed once their cue has passed.

```
                 ...video...
         [Door creaks open]
     - Is anybody home?
 ▶  12:04 ━━━━━━━━━○──────  1:34:03   🔊  CC  RTX 2x  ⚙  📁  ⛶
```

## Left to the builder

- How `Media` exposes cues to main (a `std::vector<Cue>` copy under its lock each frame is fine at subtitle sizes).
- The Vulkan texture plumbing for picture subtitles (the display already initialises the ImGui Vulkan backend).
- Tests: track naming and default choice from stream metadata, tag stripping, cue timing (start/end/no-end, seek
  clears), `.srt` beside the video found by name, the caption text layout above/below the bar in the UI checks,
  a generated video with an embedded SubRip track (FFmpeg can mux one) played through `Media` in the smoke test.
- Four bug-check agents at the end, one a security review (subtitle text and images come from untrusted files).
