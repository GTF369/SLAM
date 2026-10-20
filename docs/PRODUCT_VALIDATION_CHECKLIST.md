# Brow Design Assistant: production validation checklist

Use this list to distinguish code completion, automated tests, device testing and customer-facing readiness. A successful build does not replace a real-device or physical-transfer test.

## A. Preference recommendation

- [ ] Collect customer feedback only with specific, recorded authorization and stable anonymous customer IDs.
- [ ] Provide a clear annotation guide for ratings and the final selected design. A selected candidate and a 4–5 rating must remain distinct labels.
- [ ] Keep development, calibration and external validation customers disjoint.
- [ ] Compare against a fixed geometry-rule baseline and report case-level Top 1/Top 3 preference, ROC AUC, Brier score, calibration error and uncertainty.
- [ ] Check relevant subgroups and capture conditions, including lighting, makeup, brow density, asymmetry and different face proportions.
- [ ] Never call geometry-rule reference scores probabilities. The app only displays percentage preference estimates for a version-2 model that has separate calibration, passing external metrics and an explicit human-review approval flag.
- [ ] The current script's 30 development / 50 calibration / 100 external-customer minimums and metric thresholds are engineering screens, not statistical guarantees. Determine final sample size from the intended claims and expected error bounds.

## B. Natural eyebrow segmentation

- [ ] Build a consented annotation set, with a portion independently marked by at least two trained annotators.
- [ ] Group splits by customer so a customer's two brows or repeated photographs cannot straddle datasets.
- [ ] Report Dice, IoU, boundary F-score, false-positive/false-negative rates and region-specific performance.
- [ ] Test makeup, skin/brow color, lighting, occlusion and uneven natural brows.
- [ ] Visually verify the bundled parser's label mapping against the annotated salon set. A tensor shape check is not an accuracy test.
- [ ] Define clinically/practically acceptable contour error before test evaluation. An internal Dice threshold alone is not a release decision.

## C. 2D photo to 3D brow-arch correction

- [ ] Standardize how each of eight regions is measured, including endpoints, arc path, chord definition, tool, pose, repeated measurements and invalid samples.
- [ ] Assess within-staff and between-staff measurement repeatability.
- [ ] Use an independent 3D scan, calibrated multiview reconstruction or reliable physical measurement as reference. Do not treat single-image landmarks or relative MediaPipe z values as millimetres.
- [ ] Compute summaries after averaging repeats within each anonymous customer. Report the case-level P10–P90 empirical range separately from the bootstrap confidence interval of the mean.
- [ ] Reserve customers for independent coverage validation. Evaluate whether regional prediction ranges cover actual surface-length discrepancy at the stated rate.
- [ ] Compare the design, corrected template, printed stencil and actual transfer on skin by region and side.
- [ ] No fixed correction interval should be activated by default until independent measurement and transfer accuracy is demonstrated.

## D. Android and printer

- [ ] Test OPPO Find X7 camera permission, capture, repeated use, background/foreground, rotation, memory pressure and low-light warnings.
- [ ] Complete capture → segmentation review → edit → undo/pause → save/compare → export → print flow on target Android versions.
- [ ] Test missing/corrupt models, landmark initialization errors, extreme image sizes, print cancellation and state restoration.
- [ ] Test each intended printer model / driver / paper combination. Verify 20 mm calibration square, actual template dimensions, orientation, margins, no fit-to-page scaling and transfer alignment.
- [ ] Review logs, crash reports and accessibility/usability with practicing artists.

## E. Privacy, license and release

- [ ] Verify consent language, retention, access/deletion, and the privacy-policy URL with the operating entity and target-market counsel.
- [ ] Complete third-party model and library license review, including the EasyPortrait upstream license conditions.
- [ ] Configure a company-controlled release signing key in repository secrets only, never committed to source.
- [ ] Build and test a signed release package and its upgrade path. A debug APK is not a commercial release artifact.

## Status before in-person validation

The repository has on-device design/edit/plan workflows, model-assisted brow segmentation with manual correction, anonymous feedback export, a ranking-model training path, curvature-measurement export and grouped analysis, and print-scale correction. It does not contain real customer-trained ranking or segmentation weights. Device/printer behavior, actual stencil transfer accuracy and 3D curvature correction remain empirical acceptance items that code alone cannot establish.
