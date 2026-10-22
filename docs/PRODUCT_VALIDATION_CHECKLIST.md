# Brow Design Assistant: production validation checklist

Use this list to distinguish code completion, automated tests, device testing and customer-facing readiness. A successful build does not replace a real-device or physical-transfer test.

## A. Preference recommendation

- [ ] Collect customer feedback only with specific, recorded authorization and stable anonymous customer IDs.
- [ ] Provide a clear annotation guide for ratings and the final selected design. A selected candidate and a 4–5 rating must remain distinct labels.
- [ ] Keep development, calibration and external validation customers disjoint.
- [ ] Compare against a fixed geometry-rule baseline and report case-level Top 1/Top 3 preference, ROC AUC, Brier score, calibration error and uncertainty.
- [ ] Check relevant subgroups and capture conditions, including lighting, makeup, brow density, asymmetry and different face proportions.
- [ ] Never call geometry-rule reference scores probabilities. The app only displays percentage preference estimates for a version-3 model that has separate calibration, passing external metrics and an explicit human-review approval flag.
- [ ] The current script's 30 development / 50 calibration / 100 external-customer minimums and metric thresholds are engineering screens, not statistical guarantees. Determine final sample size from the intended claims and expected error bounds.

## B. Natural eyebrow segmentation

- [ ] Build a consented annotation set, with a portion independently marked by at least two trained annotators.
- [ ] Group splits by customer so a customer's two brows or repeated photographs cannot straddle datasets.
- [ ] Report pooled-pixel and per-ROI Dice/IoU, plus equal-weighted per-customer Dice/IoU, precision/recall, specificity, false-positive and false-negative rates. Supplement with boundary F-score and region-specific performance where available.
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
- [ ] Maintain a separate named calibration profile for each intended printer / driver / paper / setting combination. For each profile, verify the 20 mm calibration square, actual template dimensions, orientation, margins, no fit-to-page scaling and transfer alignment. The profile name is operator-managed and is not automatic printer discovery.
- [ ] Review logs, crash reports and accessibility/usability with practicing artists.

## E. Privacy, license and release

- [ ] Verify consent language, retention, access/deletion, and the privacy-policy URL with the operating entity and target-market counsel.
- [ ] Complete third-party model and library license review, including the EasyPortrait upstream license conditions.
- [ ] Configure a company-controlled release signing key in repository secrets only, never committed to source.
- [ ] Build and test a signed release package and its upgrade path. A debug APK is not a commercial release artifact.

## Automated public image smoke test

The `public-image-smoke-test` GitHub Actions job downloads two public sample portraits and the same MediaPipe / EasyPortrait model assets used by the Android package. It reports face-landmark detection, CPU model inference, class 6/7 mask coverage within the app's brow ROIs, and whether the current fallback gate accepts the model mask. It also uploads overlay previews for manual inspection.

This is a **pipeline smoke test, not an accuracy benchmark**. It has no professional eyebrow masks as ground truth, it uses Python runtimes rather than the Android ARM64 libraries, and it does not install or launch the APK. A non-empty predicted mask is not proof that it follows the true eyebrow boundary. Keep the expert-annotated customer test set and OPPO/printer tests as separate release gates.

## Status before in-person validation

The repository has on-device design/edit/plan workflows, model-assisted brow segmentation with manual correction, anonymous feedback export, a ranking-model training path, curvature-measurement export and grouped analysis, and print-scale correction. It does not contain real customer-trained ranking or segmentation weights. Device/printer behavior, actual stencil transfer accuracy and 3D curvature correction remain empirical acceptance items that code alone cannot establish.
