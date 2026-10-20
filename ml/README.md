# Data collection, ranking model, and curvature analysis

The Android app can export two separate CSV files into `Downloads/BrowDesign` on Android 10 and later:

- `brow_preference_feedback_*.csv`: anonymous case ID, candidate type and parameters, face width/height ratio, brow asymmetry proxy, rating (1–5), and whether the customer chose the design.
- `brow_curvature_calibration_*.csv`: region, measured skin-surface arc length, measured chord length, correction ratio, and timestamp.

The app does not export face photos in either data file. Obtain informed permission for collecting preference or measurement data, minimize retention, and avoid putting names, phone numbers, appointment IDs, or other identifying data into exported files.

## 1. Preference model

Install Python dependencies:

```bash
python -m pip install -r ml/requirements.txt
```

Collect observations from multiple distinct customers, then run:

```bash
python ml/train_brow_ranker.py /path/to/brow_preference_feedback.csv
```

The script rejects fewer than 30 distinct cases, keeps each `case_id` exclusively in either train or test, standardizes features using the training fold only, reports grouped holdout metrics, and writes `app/src/main/assets/brow_ranker.json`. Copy/commit that JSON into the Android source tree and rebuild to enable local model inference. If the model file is absent or invalid, the app uses the explicit geometry heuristic fallback.

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

## 4. Limitations

The app currently generates the brow outline from a small set of facial landmarks and adjustable parameters. It does not yet segment the client's original eyebrow hairs, does not reconstruct a measured 3D brow surface, and does not automatically warp the print template to a per-client 3D surface. The preference model is only activated after real data are collected and a validated JSON model is added. No synthetic model is included in this repository.
