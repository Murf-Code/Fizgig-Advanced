# Blur NSFW samples (opt-in)

In Samples → Previews so far, switch on **Blur NSFW samples** (off by default).
Only generated still-image sample thumbnails, run-history samples, their lightbox
and the run visualiser are marked for scanning. Dataset images, Caption Workbench,
Image Prep and original files are unchanged. Video/audio samples and exported files
are not screened. Click an obscured picture to reveal it in the current page session.

Uses the existing Fizgig CPU ONNX body-part detector framework. Install **Anime
NSFW All-in-One v5.0** through the existing Image Prep body-detector setup
(see docs/FACE_BACKENDS.md). Optional booru_yolo supplements detection of
illustrated/furry anatomy. No automatic downloads, image uploads or GPU inference.
Missing or incompatible detectors and scan errors leave the sample obscured,
with a click-to-reveal option. Detector accuracy varies by style and pose;
this is not guaranteed to identify every NSFW image, sexual content or violence.
The image is only visually obscured, not access-controlled or altered.
