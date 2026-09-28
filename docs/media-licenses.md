# Media licenses — demo, site and video

Everything that appears in the application demo, on the project site or in the demonstration video, with its
source, license and the evidence checked. Verified on **2026-09-27**. Anything whose permitted use could not be
confirmed would have been replaced; nothing needed replacing.

## Background music (video only)

| Field | Value |
|---|---|
| Track | **Dreamer** |
| Creator | **Kevin MacLeod** |
| Publisher / download source | incompetech.com — https://incompetech.com/music/royalty-free/mp3-royaltyfree/Dreamer.mp3 |
| Catalogue entry | incompetech `pieces.json`: title "Dreamer", ISRC **USUAN1600043**, 3:24, piano and percussion, feel "Calming, Relaxed", uploaded 2016-08-06 |
| Downloaded file | `Dreamer.mp3`, 6,532,428 bytes, 256 kb/s, sha256 `615e65c9ca131c11f8e12970644ed74ef822ec4f643b1d9d0085a223600bcdfb`; ID3: title "Dreamer", artist "Kevin MacLeod", comment "ISRC: USUAN1600043". Kept in the git-ignored `.cache/music/`, **not committed** |
| License | **Creative Commons Attribution 4.0 International (CC BY 4.0)** — https://creativecommons.org/licenses/by/4.0/ |
| Required attribution (publisher's format) | Dreamer Kevin MacLeod (incompetech.com)<br>Licensed under Creative Commons: By Attribution 4.0<br>https://creativecommons.org/licenses/by/4.0/ |
| Changes (must be disclosed) | Excerpt from the start of the track trimmed to the video length, lowered to background level and automatically ducked (side-chain compression) under the narration, fade-in and fade-out, re-encoded to AAC and mixed with the narration. The narration and everything else in the video are this project's; the music is Kevin MacLeod's. |

**Evidence that this use is permitted**

- incompetech FAQ (https://incompetech.com/music/royalty-free/faq.html): music may be used in videos, including
  monetised ones — "Be sure to credit me"; the credit must be placed so "a person who wants to know where the music
  came from should have no difficulty in finding it" (video description or in the video); the credit format above
  is quoted from the FAQ, which names the license "Creative Commons: By Attribution 4.0"; edits are allowed but the
  credits "MUST make it clear … which parts are yours, and which parts are mine".
- incompetech licensing page (https://incompetech.com/music/royalty-free/licenses/): the free option "Requires that you
  credit the music".
- CC BY 4.0 legal code: share and adapt for any purpose, provided appropriate credit, a link to the license, and an
  indication of changes.

**Residual risk.** The track page on the publisher's licensing platform (`incompetech.filmmusic.io/song/3676-dreamer/`)
redirected to `ende.app/song/3676-dreamer/`, which returned 404 on the verification date. The license was therefore
verified from the publisher's FAQ, licensing page, catalogue data and the file's own tags. Automated content matching on
video platforms can flag CC music; the credit above is the basis for a dispute. Only the mixed video is distributed.

**Where the credit appears:** end card of the video, the video's caption/description block on the site, the README
Credits section, and this file.

## Narration and demo voices

Synthetic speech generated locally with **Kokoro-82M v1.0** (hexgrad; Apache-2.0 weights,
https://huggingface.co/hexgrad/Kokoro-82M) via **kokoro-onnx** 0.6.1 (MIT, https://github.com/thewh1teagle/kokoro-onnx).
Apache-2.0 permits use and redistribution of the model; generated audio carries no additional restriction from the
model license. The model card credits CC BY training data (Koniwa, CC BY 3.0; SIWIS, CC BY 4.0); we repeat that credit.
The narration is labelled as a synthetic voice in the video and on the site. No real person's voice is used.

## Photos

Public-domain US government photos (FEMA, NPS); full table with source links and original-file hashes in
[`demo/CREDITS.md`](../demo/CREDITS.md). They show real, unrelated past events and are paired with fictional reports only
for the demo — the video says so on screen.

## Screenshots

All application screenshots are of this project's own software running locally with the demo inputs above. No
CrisisMMD tweet or image appears in any screenshot, on the site or in the video.

## Fonts and code on the site

The site uses the portfolio's font choice (Inter, JetBrains Mono — SIL Open Font License) loaded from Google Fonts, and
its visual tokens, which belong to the same owner's portfolio.
