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

The repository also contains an experimental conditional-choice trainer, `ml/train_brow_choice_ranker.py`, with a synthetic-only software check in `ml/simulate_brow_choice_sandbox.py`. It is **not loaded by the current APK**. It expects a specially prepared CSV with a stable `session_id`, exactly three candidate rows (natural, soft, defined) per session, and exactly one selected candidate. The current APK's feedback export does not provide `session_id`, and Android currently loads only the approved version-3 `brow_ranker.json` model described above. Do not copy the experimental choice model into app assets or treat it as deployed. Integrating it would require a matching three-candidate feedback session workflow, app-side inference and separate real-customer validation.

## 2. What public models and datasets can and cannot provide

The public eyebrow-preference study [*Eyebrow Shape Preference Across Age, Gender, and Self-reported Ethnic Group*](https://doi.org/10.1097/dss.0000000000003679) reports survey results from 652 respondents, but the publication itself is not a ready-to-load Android model or a verified case-level training CSV containing each face, candidate parameters, and ratings. A periorbital segmentation dataset can help label or detect eyebrow pixels, but mask labels do not tell us whether a particular customer likes a candidate. I did not find a publicly downloadable, clearly licensed, customer-level preference dataset that can legitimately serve as this app's calibrated model.

Therefore, do not convert published group averages, geometric rules, synthetic rows, or generic face-attractiveness scores into an advertised probability of a customer's brow preference. The application can emit calibrated probabilities only after the candidate ranking model has been trained and independently checked with consented ratings from the intended feedback process. Until then, keep the transparent geometric reference score.

## 3. Public-image engineering stress test

A reproducible, non-training experiment is available at [docs/PUBLIC_IMAGE_BROW_BENCHMARK.md](../docs/PUBLIC_IMAGE_BROW_BENCHMARK.md) and `ml/public_image_brow_benchmark.py`. It applies the current rule-based ranker to five properly attributed Wikimedia Commons portraits, repeats ranking over fixed image perturbations, records MediaPipe relative-z sensitivity, and sweeps theoretical circular arc/chord factors. Its normalized score weights are **not preference probabilities**, and the image set has no customer-preference labels. The experiment is useful for software robustness and for spotting rule bias only.

## 4. Commercial data licensing and privacy

Do not assume a public face dataset can be used commercially. The official CelebAMask-HQ agreement, for example, limits its images and derived data to non-commercial research and prohibits commercial exploitation. See the [official dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html). Use it for this product only if the rights holder grants separate written permission.

Prefer consented, purpose-limited salon data with documented authorization for training, validation, deployment and the target market. Keep a dataset register with source, permission/license, permitted uses, collection date, annotation version, retention and deletion status. Downloadable does not mean commercially licensed.

## 5. Curvature measurement and interval analysis

Run:

```bash
python ml/analyze_curvature.py brow_curvature_calibration.csv --output curvature_summary.csv
```

For each region, the script first computes the mean repeated measurements within each anonymous customer. Population summaries and the P10–P90 empirical interval are then calculated from those **case-level means**, so a customer with extra repeated measurements cannot dominate the interval. It also reports the number of raw measurements, number of cases, observed case-level range, bootstrap 95% confidence interval for the population mean, and case-level radius summaries when available.

The P10–P90 interval is a descriptive central range of the observed customers. It is not a guaranteed anatomical bound, an individual prediction interval, or by itself a correction recommendation. The bootstrap interval refers to uncertainty in the population mean, not the range that will contain 90% of individual customers. Legacy CSV files without a `case_id` cannot group repeated measurements and are explicitly warned about.

The directly measured ratio is

`k = measured skin-surface arc length / measured chord length`.

For an ideal circular arc with chord (c) and radius (R), (s = 2R\arcsin(c/(2R))) and (k=s/c). Real brow ridges are not perfect circular arcs. A single frontal photograph or MediaPipe relative-z values do not provide true millimetre-scale 3D geometry.

Follow [CURVATURE_PROTOCOL.md](CURVATURE_PROTOCOL.md): consistent endpoints, repeat measurements, customer-level grouping, staff-to-staff repeatability checks, and an independent 3D or reliable physical reference. Before any factor is embedded in normal printing, validate the expected coverage of the regional interval and measure actual stencil transfer errors on unseen customers.

## 6. Brow-region segmentation

The Android app supports three segmentation paths:

- **Customer-trained eyebrow model**: when `app/src/main/assets/brow_segmentation.tflite` exists with float32 input `[1,256,256,3]` and output `[1,256,256,1]`, it is preferred.
- **EasyPortrait face parsing model**: the Android CI downloads a compact 19-channel ONNX model, using the inferred common 19-class label map. Its exact eyebrow mapping still needs visual validation against professionally annotated salon photos.
- **Adaptive contrast fallback**: if inference is unavailable or weak, the app estimates brow pixels around landmarks. It remains manually editable and is not a precise segmentation guarantee.

No customer-specific segmentation model is shipped without a consented, professionally annotated dataset. See [MODEL_PROVENANCE.md](../docs/MODEL_PROVENANCE.md) for model source/license caveats and [PRODUCT_VALIDATION_CHECKLIST.md](../docs/PRODUCT_VALIDATION_CHECKLIST.md) for the test gates.

## 7. Remaining production gates

- Real, consented and professionally reviewed customer feedback and brow-mask annotations.
- Separate calibration and independent validation sets with disjoint customer IDs.
- Measured 3D/physical reference and stencil-transfer validation by region.
- OPPO Find X7 and each target printer tested in person, including 20 mm scale accuracy, no auto-fit setting, page margins, orientation and cancellation.
- Device performance/crash review, privacy-policy review, third-party license review and a signed release build.


## Conditional choice model from real customer decisions

The Android app now includes **三款选择采集** and **导出选择会话**. Each session renders the same customer's photo with the fixed natural / soft / defined candidates, records exactly one customer-selected candidate, optionally records a 1–5 satisfaction rating for each candidate, and stores a stable anonymous `case_id` plus a unique `session_id`. The exported CSV contains numerical measurements and candidate parameters, not the portrait. Obtain explicit, purpose-specific consent before exporting or training.

Do not label an LLM compatibility score, a geometric score, or a public-portrait heuristic as a customer choice. The selected option must come from the customer's own decision. Optional ratings are useful secondary outcomes but are not substituted for the selected-candidate label.

### 1. Export and split the real feedback

Export from Android into `Downloads/BrowDesign/brow_choice_feedback_*.csv`. Copy the file into your local analysis environment and run:

```bash
python ml/split_choice_cases.py brow_choice_feedback.csv --output-dir ml/data_splits/choice
```

The splitter validates that every session has exactly three rows, exactly one of each style, and exactly one selected row. It partitions by anonymous customer ID, never by candidate row or session, so repeated visits from one person remain in a single split. Default proportions are 40% development, 20% calibration and 40% external validation. It writes `split_report.json` with per-split row, session and distinct-customer counts.

### 2. Train and calibrate

```bash
python ml/train_brow_choice_ranker.py ml/data_splits/choice/development.csv \
  --calibration-csv ml/data_splits/choice/calibration.csv \
  --external-validation-csv ml/data_splits/choice/external_validation.csv \
  --output app/src/main/assets/brow_choice_ranker.json
```

The conditional choice model is a regularized utility model that learns from actual customer-selected candidates and interactions between face measurements, candidate parameters and the **customer-stated** style intent. Temperature scaling is fitted on the separate calibration customers. External evaluation is kept separate and reports top-1 accuracy and its paired gain over the geometry baseline, negative log likelihood, multiclass Brier score and calibration error with customer-cluster bootstrap uncertainty.

Current minimum engineering gates are 30 development customers, 50 separate calibration customers and 100 separate external-validation customers. At a 40/20/40 split, plan on at least about 250 distinct consenting customers, usually with at least one complete three-candidate session each. Multiple sessions from one person do not count as multiple independent customers. These thresholds are early screening gates, not a guarantee of statistical sufficiency. Prefer a **prospective future-customer validation cohort** collected only after model development and calibration are frozen.

Only after checking the report, target-population representativeness, subgroup errors, the baseline comparison and the independent cohort may a human reviewer mark the artifact releasable:

```bash
python ml/train_brow_choice_ranker.py ml/data_splits/choice/development.csv \
  --calibration-csv ml/data_splits/choice/calibration.csv \
  --external-validation-csv ml/data_splits/choice/external_validation.csv \
  --mark-reviewed-after-human-review \
  --output app/src/main/assets/brow_choice_ranker.json
```

Do not run the review flag just to produce percentages. The Android app loads `brow_choice_ranker.json` only when it is version 2 or newer, declares `model_type=conditional_choice`, includes temperature calibration, and has `release_approved: true`. If no such model exists, the app records the real choice but deliberately displays no probability. Once approved, probabilities are **conditional on the three rendered candidates and the feedback protocol**; they are not objective attractiveness, universal suitability or a guaranteed treatment outcome.

### 3. External LLM review is optional and not training truth

The app includes an optional OpenAI-compatible Chat Completions adapter. At runtime, the user enters a supported endpoint and model ID and explicitly confirms customer consent. It sends only numerical ratios, the three candidate parameters and the stated design intent; it does not send the portrait or anonymous case/session IDs. LLM output is an **uncalibrated compatibility opinion**, never the ground-truth label and never a customer preference probability. It must not be silently mixed into the customer-choice training CSV.

Direct API keys embedded in a mobile app can be extracted. Prefer a self-controlled HTTPS proxy that holds provider credentials server-side, uses usage limits and does not log faces or identifiers. The in-app entry is intended for controlled personal testing rather than a secure multi-user deployment.

### 4. Known evidence limits

- The public-image test is a robustness experiment and has no customer preference labels.
- Generated synthetic choice rows validate software only and must never train a production model.
- A properly calibrated model still depends on how candidates are rendered, the population collected and whether customers make unpressured choices.
- Ratings can be affected by artist presentation, lighting, makeup, novelty and the order of candidates. Use randomized display order in a later prospective study while preserving the style-to-row mapping, and document the protocol.
- The present app records the choice among three fixed styles. It does not yet prove probabilities for arbitrary thickness / length / arch edits outside the trained candidate distribution.
