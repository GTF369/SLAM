# Data collection, ranking model, and curvature analysis

The Android app exports anonymous preference feedback and separate curvature measurements to `Downloads/BrowDesign` on Android 10 and later. The preference CSV records candidate type/parameters, face geometry ratios, reviewed native-brow mask area/asymmetry features, whether mask features were reviewed and whether the native contour was selected, a 1–5 customer rating and the selected flag. The curvature CSV records region, measured skin-surface arc length, chord length, correction ratio and timestamp. No face photo is included in these CSVs.

Collect and export such data only after appropriate consent for the stated use. Keep a data register and avoid names, phone numbers, appointment numbers and any identifiers that can reveal a customer. Use the app's **匿名案例编号** control to reuse the same random pseudonymous code for all candidate designs, retakes and follow-up measurements from the same customer when those records are authorized for the same study. Generate a new code for a different customer. If you use a new code on every retake for the same customer, grouped evaluation can leak repeated-customer preferences across splits. Exported case IDs must be stable per customer but must not identify the person.

## 1. Preference model: development, calibration, and external test

The app's built-in geometry score is a transparent prototype, not a learned probability. The training pipeline below intentionally produces a non-releasable model unless it receives separate data for calibration and external validation. Synthetic test fixtures in `ml/tests` only test software behavior. They are not customer data and must never be used to build the app model.

Install dependencies:

```bash
python -m pip install -r ml/requirements.txt
```

For a convenience split of one collected export, run:

```bash
python ml/split_preference_cases.py brow_preference_feedback.csv --output-dir ml/data_splits
```

This groups all candidate rows from the same anonymous customer in one file. It is a reproducible customer-level holdout from the same collection, not a prospective future-cohort study. For stronger evidence, replace the external split with feedback from new customers collected after the model-development and calibration process has been frozen.

Prepare three CSV files containing mutually disjoint anonymous customers:

- **Development CSV**: used to train model coefficients and for an internal case-grouped 80/20 diagnostic split.
- **Calibration CSV**: used only to fit a Platt scaling transform to turn raw logits into a calibrated estimate for a high customer rating.
- **External validation CSV**: used only once for a final estimate of ranking/discrimination and calibration quality. It must not be used to tune the model.

```bash
python ml/train_brow_ranker.py ml/data_splits/development.csv \
  --calibration-csv ml/data_splits/calibration.csv \
  --external-validation-csv ml/data_splits/external_validation.csv \
  --output app/src/main/assets/brow_ranker.json
```

The base training refuses fewer than 30 development cases. The calibration set must contain at least 50 distinct customers and the external validation set at least 100 distinct customers. These are early engineering gates, not a statistical guarantee that the model is production ready. Same-customer rows are kept within one dataset, and the three dataset case IDs must not overlap. External validation also compares the candidate against the app's current geometry-rule baseline, with a paired customer-cluster bootstrap interval for AUC and the AUC gain. For release, the 95% AUC interval must be above chance and the 95% interval for the AUC gain over the geometry baseline must be entirely positive.

The output JSON is a **candidate only**. Review the grouped holdout and external metrics, confirm that the validation population resembles intended salons/customers, investigate errors and subgroup performance, inspect the customer-cluster bootstrap interval, and have a qualified reviewer sign off. Only then may you explicitly mark the candidate reviewed:

```bash
python ml/train_brow_ranker.py development.csv \
  --calibration-csv calibration.csv \
  --external-validation-csv external_validation.csv \
  --mark-reviewed-after-human-review \
  --output app/src/main/assets/brow_ranker.json
```

That command also requires the external data to meet preliminary engineering gates: ROC AUC at least 0.60, the lower 95% customer-cluster bootstrap AUC bound above 0.50, the lower 95% bound for AUC improvement over the geometry-rule baseline above zero, Brier score better than the constant-prevalence baseline and 10-bin expected calibration error at most 0.10. Those cutoffs are screening rules, not industry standards and not proof of fairness or aesthetic correctness. A model JSON must not be marked reviewed just to enable the app.

Only a version-3 JSON explicitly marked `release_approved: true`, with a separate Platt calibrator and passing external metrics, will be loaded by Android. The live percentage estimates the probability that a candidate receives a 4–5 rating under the collected feedback process. It does **not** mean objective beauty, universal suitability, or a guaranteed treatment result. When no reviewed model is present, the app continues to use geometric reference scores and clearly labels them as such.

The model now includes four extra inputs: reviewed native-brow area fraction, left/right mask-area imbalance, mask-reviewed availability, and whether the native contour was used for this candidate. Mask statistics are only collected when the artist explicitly confirms the detected mask via “完成修正” or selects “以原生眉形作底稿”. The target label is rating 4–5 versus 1–3. The `selected` field remains available for audit but is not an input to the model. A selected candidate and a high rating are related but not interchangeable. Before commercial use, evaluate top-k selection, performance across face shapes and capture conditions, subgroup errors, confidence intervals, data drift and the way ratings were collected.

## 2. Commercial data licensing and privacy

Do not assume a public face dataset can be used commercially. The official CelebAMask-HQ agreement, for example, limits its images and derived data to non-commercial research and prohibits commercial exploitation. See the [official dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html). Use it for this product only if the rights holder grants separate written permission.

Prefer consented, purpose-limited salon data with documented authorization for training, validation, deployment and the target market. Keep a dataset register with source, permission/license, permitted uses, collection date, annotation version, retention and deletion status. Downloadable does not mean commercially licensed.

## 3. Curvature measurement and interval analysis

Run:

```bash
python ml/analyze_curvature.py brow_curvature_calibration.csv --output curvature_summary.csv
```

For each region, the script first computes the mean repeated measurements within each anonymous customer. Population summaries and the P10–P90 empirical interval are then calculated from those **case-level means**, so a customer with extra repeated measurements cannot dominate the interval. It also reports the number of raw measurements, number of cases, observed case-level range, bootstrap 95% confidence interval for the population mean, and case-level radius summaries when available.

For physical print-to-skin validation, the app also exports a separate transfer-validation CSV after the operator measures the printed template and its transferred placement. Analyze it with:

```bash
python ml/analyze_stencil_transfer.py brow_transfer_validation.csv --output stencil_transfer_summary.csv
```

Rows are grouped by printer profile, eyebrow region, and curvature method. Repeated observations are averaged within anonymous customer inside each method before summaries and bootstrap intervals are computed, so the no-correction baseline is never pooled with measured-curvature trials. The report includes signed and absolute template-length error, P90 absolute error, endpoint-placement error, sample coverage and customer counts. To run a predeclared engineering screen, supply both tolerances and the minimum case count, for example:

```bash
python ml/analyze_stencil_transfer.py brow_transfer_validation.csv \\
  --min-cases 30 \\
  --max-p90-abs-length-error-mm 1.0 \\
  --max-p90-endpoint-error-mm 1.5 \\
  --output stencil_transfer_summary.csv
```

The tolerance values above are illustrative command-line examples, **not clinically validated limits or universal standards**. Set permissible error margins with the practicing artist and intended workflow before collecting validation data. A screening pass only means the group is eligible for human review. The script never produces or applies an automatic correction coefficient, and it cannot establish 3D surface accuracy by itself.

The P10–P90 interval is a descriptive central range of the observed customers. It is not a guaranteed anatomical bound, an individual prediction interval, or by itself a correction recommendation. The bootstrap interval refers to uncertainty in the population mean, not the range that will contain 90% of individual customers. Legacy CSV files without a `case_id` cannot group repeated measurements and are explicitly warned about.

The directly measured ratio is

`k = measured skin-surface arc length / measured chord length`.

For an ideal circular arc with chord (c) and radius (R), (s = 2R\arcsin(c/(2R))) and (k=s/c). Real brow ridges are not perfect circular arcs. A single frontal photograph or MediaPipe relative-z values do not provide true millimetre-scale 3D geometry.

Follow [CURVATURE_PROTOCOL.md](CURVATURE_PROTOCOL.md): consistent endpoints, repeat measurements, customer-level grouping, staff-to-staff repeatability checks, and an independent 3D or reliable physical reference. Before any factor is embedded in normal printing, validate the expected coverage of the regional interval and measure actual stencil transfer errors on unseen customers.

## 4. Brow-region segmentation

The eyebrow ROI training script reports pooled-pixel metrics as well as ROI-macro and equal-weighted customer-macro Dice/IoU and error rates. See `ml/segmentation_metrics.py`. Interpret the pooled score together with customer-macro results because multiple photos or left/right ROIs from one customer must not dominate model selection. These remain internal grouped-test metrics and are not a substitute for a prospective external salon cohort.


The Android app supports three segmentation paths:

- **Customer-trained eyebrow model**: when `app/src/main/assets/brow_segmentation.tflite` exists with float32 input `[1,256,256,3]` and output `[1,256,256,1]`, it is preferred.
- **EasyPortrait face parsing model**: the Android CI downloads the current 19-channel ONNX export. Public-image class contact sheets exposed that its brow channels do not match the assumed CelebAMask-HQ 19-class order. The Android engine and smoke test now use channels 2 and 3, as visually aligned with brows in the tested images. This is a label-index fix, not segmentation accuracy validation; professional brow-mask annotations and an independent client cohort remain necessary.
- **Adaptive contrast fallback**: if inference is unavailable or weak, the app estimates brow pixels around landmarks. It remains manually editable and is not a precise segmentation guarantee.

No customer-specific segmentation model is shipped without a consented, professionally annotated dataset. See [MODEL_PROVENANCE.md](../docs/MODEL_PROVENANCE.md) for model source/license caveats and [PRODUCT_VALIDATION_CHECKLIST.md](../docs/PRODUCT_VALIDATION_CHECKLIST.md) for the test gates.

## 5. Remaining production gates

- Real, consented and professionally reviewed customer feedback and brow-mask annotations.
- Separate calibration and independent validation sets with disjoint customer IDs.
- Measured 3D/physical reference and stencil-transfer validation by region.
- OPPO Find X7 and each target printer tested in person, including 20 mm scale accuracy, no auto-fit setting, page margins, orientation and cancellation.
- Device performance/crash review, privacy-policy review, third-party license review and a signed release build.
