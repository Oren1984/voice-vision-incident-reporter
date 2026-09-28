# Demo media credits

All demo inputs are **illustrative**. The spoken reports are fictional texts written for this project and
voiced by a synthetic TTS model. The photos show real, unrelated past events; each report–photo pairing was
invented to exercise the workflow and is **not** real incident data. Verified on **2026-09-27**.

## Photos (public domain, US federal government works)

| File in `demo/images/` | Source file | Creator | License (as tagged on Commons) | Commons SHA-1 of original |
|---|---|---|---|---|
| `flood_street_fema_34508.jpg` | [FEMA - 34508 - Flooded street in Missouri.jpg](https://commons.wikimedia.org/wiki/File:FEMA_-_34508_-_Flooded_street_in_Missouri.jpg) — "Pacific, MO, March 22, 2008 — Water remains on streets in neighborhoods near the Meramec River." | Jocelyn Augustino / FEMA | Public domain — `PD-USGov-FEMA` (work of a FEMA employee in the course of official duties) | `55d8e14b1ee60ee4792dd97f2056cc3d36e82297` |
| `collapsed_building_fema_northridge.jpg` | [Partially collapsed apartment building, 1994 Northridge Earthquake.jpg](https://commons.wikimedia.org/wiki/File:Partially_collapsed_apartment_building,_1994_Northridge_Earthquake.jpg) | FEMA News Photo | Public domain — `PD-USGov-FEMA` | `c889e774673703c113713b440563eca911368595` |
| `brush_fire_nps_horne_2021.jpg` | [Suppression efforts from the 2021 Horne Fire.](https://commons.wikimedia.org/wiki/File:Suppression_efforts_from_the_2021_Horne_Fire._(e082090e-2335-42bd-9d5c-4290e59ff648).jpg) (NPGallery) | NPS Photo | Public domain — `PD-USGov-NPS` | `d60d3b0dd0efb47edc1d095a3669aa43cf7600bb` |
| `flood_street_fema_34508_blurred_dark.jpg` | derived from the FEMA flood photo above | this project | public domain source; our edit: Gaussian blur (radius 9) and brightness × 0.25, to simulate an unclear photo | — |

Edits to all photos: downscaled to 1600 px on the long side and re-encoded as JPEG (quality 88). No
attribution is legally required for US federal public-domain works; credit is given as good practice. Public
domain status covers copyright only — the photos do not imply endorsement by FEMA or the NPS.

## Voices

`demo/audio/*.wav` were generated with **Kokoro-82M v1.0** (hexgrad, Apache-2.0 weights;
https://huggingface.co/hexgrad/Kokoro-82M) through **kokoro-onnx** 0.6.1 (MIT; https://github.com/thewh1teagle/kokoro-onnx),
voices `af_heart`, `am_michael`, `bf_emma`, `bm_george`. The Kokoro model card lists CC BY training data it
credits: Koniwa (CC BY 3.0) and SIWIS (CC BY 4.0). `s4_flood_report_noisy.wav` adds synthetic white noise
(0 dB SNR, seed 2026). Regenerate with `python tools/make_demo_media.py`.

No recording of a real person is used anywhere in this repository.
