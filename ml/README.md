# Data collection, ranking model, and curvature analysis

The Android app can export two separate CSV files into `Downloads/BrowDesign` on Android 10 and later:

- `brow_preference_feedback_*.csv`: anonymous case ID, candidate type and parameters, face width/height ratio, brow asymmetry proxy, rating (1–5), and whether the customer chose the design.
- `brow_curvature_calibration_*.csv`: region, measured skin-surface arc length, measured chord length, correction ratio, and timestamp.

The app does not export face photos in either data file. Obtain informed permission for collecting preference or measurement data, minimize retention, and avoid putting names, phone numbers, appointment IDs, or other identifying data into exported files.

## Commercial data licensing

Do not assume a public face dataset can be used commercially. For example, the official CelebAMask-HQ agreement explicitly limits the images and derived data to non-commercial research and prohibits commercial exploitation. See the [official dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html). Do not use it to train this product unless the rights holder grants a separate written commercial license.

For product training, prefer:
- consented, purpose-limited salon images and annotations with documented authorization for model training and evaluation;
- a separately licensed dataset whose written terms expressly allow commercial model training, derivation of weights, deployment, and the target regions;
- a data processing and retention policy matching the target market.

Keep a dataset register with source, license/consent basis, permitted uses, collection date, annotation version, permitted retention, and deletion status. A dataset being downloadable or marked "open" is not proof that commercial use is permitted.

## 1. Preference model

Install Python dependencies:

```bash
python -m pip install -r ml/requirements.txt
```

Collect observations from multiple distinct customers, then run:

```bash
python ml/train_brow_ranker.py /path/to/brow_preference_feedback.csv
```

The script rejects fewer than 30 distinct cases and keeps each `case_id` exclusively in either train or test. It standardizes features on the training fold only for the grouped 80/20 holdout assessment, reports holdout metrics, then refits the deployable candidate using all available consented cases and writes `app/src/main/assets/brow_ranker.json`. The reported holdout metrics describe the assessment model, not the final all-data fit. Copy/commit the JSON into the Android source tree and rebuild to enable local model inference. If the model file is absent or invalid, the app uses the explicit geometry heuristic fallback. A prospective independent set is still needed before relying on the model in client work.

Model input feature order is fixed and must remain aligned between Python and Kotlin:

1. `face_width_height_ratio`
2. `interocular_face_width_ratio`
3. `jaw_face_width_ratio`
4. `brow_eye_distance_ratio`
5. `brow_asymmetry_ratio`
6. `eye_line_tilt`
7. `thickness`
8. `length`
9. `arch`
10. `style_soft` (0/1)
11. `style_defined` (0/1)

Rating 4–5 is considered a positive preference label and rating 1–3 a negative label. `selected` is retained for audit/analysis but is not used as a model input or target. This is an exploratory preference model, not a model of clinical outcome and not proof that a particular brow shape is objectively best. A minimum of 30 cases is only a software guardrail. For commercial claims, gather a substantially larger dataset, evaluate by salon and demographic strata where appropriate, and conduct a prospective external test before deployment.

## 2. Curvature data analysis

Run:

```bash
python ml/analyze_curvature.py /path/to/brow_curvature_calibration.csv --output curvature_summary.csv
```

The report gives region-level measurement count, distinct anonymous case count, mean, median, 10th/90th percentiles, observed range, a case-clustered bootstrap confidence interval for the mean, and estimated circular-arc radii when available. The P10–P90 measurement interval is descriptive, not an anatomical guarantee or individual prediction interval. The estimated radius assumes a local circular arc and is not a three-dimensional reconstruction. Establish a standardized measurement protocol, repeatability limits, and an independent validation set before embedding any region-specific correction factors in clinical practice.

## 3. Measurement geometry

The directly measured ratio is

`k = measured surface arc length / measured chord length`.

For an ideal circular arc, if the chord (c) and radius (R) are known, then:

`s = 2 R arcsin(c / (2 R))`

and `k = s / c`. Real brow ridges are not perfect circular arcs, and a single frontal photograph does not identify the depth or curvature. Do not infer a universal (R) or universal correction interval from the photo alone.


## 4. Brow-region segmentation

The Android app has two modes:

- **Adaptive contrast pre-segmentation** is available immediately, without a trained model. It estimates dark eyebrow pixels from local luminance around the face-landmark brow ROI and provides a sensitivity slider, an overlay, and touch brushes to add/remove pixels. It is a convenient starting mask, not a validated precise segmenter.
- **Optional learned segmentation** is activated when `app/src/main/assets/brow_segmentation.tflite` is present and the model has the expected float32 input `[1,256,256,3]` (RGB normalized to 0–1) and output `[1,256,256,1]` (sigmoid probabilities). The app falls back safely if this model is absent or incompatible.

### Collect and annotate data

1. Obtain specific consent to export and annotate facial photos. In the app, use **导出原始标注照片** only when the customer has agreed to this separate training use.
2. Name source images by anonymous case, for example `CASE001_left.jpg` and `CASE001_right.jpg`. Do not use names, appointment numbers or phone numbers.
3. Install optional tools with `python -m pip install -r ml/requirements-segmentation.txt`.
4. Run `python ml/annotate_brow_masks.py --images /local/consented/photos --output /local/brow_dataset`. Select each brow ROI with margins similar to the app's landmark ROI, then outline visible eyebrow hair/body with the polygon editor. Review masks manually.
5. The tool generates `images/`, `masks/` and `manifest.csv`. The manifest has `image,mask,case_id`. Same-client images must share one case ID to prevent data leakage between training and validation.

### Train and export

```bash
python ml/train_brow_segmentation.py --dataset /local/brow_dataset --output ml/artifacts/brow_segmentation_candidate.tflite --metrics ml/brow_segmentation_metrics.json
```

By default the output is a candidate model and is **not** copied into the Android app. After reviewing masks and holdout examples, an initial internal screening step can be run with:

```bash
python ml/train_brow_segmentation.py --dataset /local/brow_dataset --output ml/artifacts/brow_segmentation_candidate.tflite --metrics ml/brow_segmentation_metrics.json --promote-to-app --min-test-dice 0.70
```

The second command only copies the model to app assets if the untouched grouped test Dice is at least 0.70. The model-selection validation split is separate from this test split, so the final test cases are not used for early stopping. Passing this internal gate still does not mean the model is commercially validated. The training script uses a compact U-Net, binary-cross-entropy plus Dice loss, aligned paired augmentations, and three case-grouped partitions for training, early-stopping validation and final internal testing. It refuses to train fewer than 50 distinct cases by default. This is only an initial exploratory floor, not a sufficient commercial sample-size claim. Review the masks and test examples, including false positives and false negatives, before committing a model. Validate on independently collected clients, skin tones, makeup, lighting and camera conditions and then on the target Android device.

No trained brow-segmentation model is shipped in this repository because no authorized, professionally annotated dataset was supplied. The application therefore uses the adaptive pre-segmentation fallback until a compatible trained model is added.

## 5. Regional curvature correction in printing

The print dialog can apply experimental regional length factors only when all eight regions for the current anonymous case have been measured. The factors are per-region medians of `arc_mm / chord_mm` and are capped at 1.4 when applied to adjacent landmark segments. The print page marks the correction as experimental.

This correction is a regional length adjustment. It is not full 3D reconstruction or a validated physical unwrapping of an individual's brow ridge. The measurement protocol and independent validation requirements are in [CURVATURE_PROTOCOL.md](CURVATURE_PROTOCOL.md).

## 6. Remaining production gates

- No trained ranking model is present until real feedback data is collected and a model JSON is generated.
- No trained segmentation model is present until authorized images and professionally reviewed masks are collected.
- The experimental regional correction must be evaluated against a measured surface model and physical stencil transfer, with print-scale checks on each printer model.
- The Android app must complete OPPO Find X7 and target-printer field tests, crash/memory checks, and a signed release build.
