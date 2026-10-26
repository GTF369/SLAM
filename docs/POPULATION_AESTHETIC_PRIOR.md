# Optional population aesthetic prior (not bundled by default)

The optional Android asset `brow_aesthetic_resnet50.onnx` can be exported from
[evanlyhf/scut-fbp5500-beauty](https://huggingface.co/evanlyhf/scut-fbp5500-beauty),
a ResNet-50 regressor trained on SCUT-FBP5500 human facial-attractiveness ratings.
**It is not downloaded or packaged in the standard APK build.** The standard app therefore
falls back to its geometric reference score unless an authorized optional asset is supplied.
The model-card inference transform is a 224×224 resize, RGB values scaled to [0,1],
then ImageNet normalization with means [0.485, 0.456, 0.406] and standard deviations
[0.229, 0.224, 0.225]. The app averages original and horizontally-flipped inference.

**Interpretation limit:** the model predicts a whole-face population rating, not eyebrow
compatibility, not an individual customer's choice probability, and not an outcome of a
permanent-makeup procedure. Comparing three brows rendered onto one face may give a
useful weak ranking signal, but small differences can be noise and the model was not
trained on paired eyebrow designs. The app therefore labels output as a *群体审美参考分*
(1–5) and never turns it into a percentage or claims it is the user's personal probability.

## Licence and intended use

The Hugging Face model card lists licence as `other`, limits intended use to research and
education, and notes that SCUT-FBP5500 dataset terms apply. The official dataset repository
says the dataset is for **non-commercial research only**. The model is not shipped by default
because personal or non-commercial use is not automatically the same as permitted research
or educational use. Do not download, convert, embed, or redistribute it unless the intended use
is explicitly allowed by the current upstream terms or the necessary written permission has
been obtained. The app's normal fit-estimation workflow does not require this model.
Sources:
- https://huggingface.co/evanlyhf/scut-fbp5500-beauty
- https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

## Export

The standard GitHub Actions APK build does not download this checkpoint or generate the ONNX
asset. Only after confirming that the exact intended use is permitted, a separately authorized
local research build can generate it. Install CPU PyTorch 2.6, torchvision 0.21, timm 1.0.15,
ONNX 1.17, and ONNX Runtime 1.23.2, then run:

```bash
python ml/export_brow_aesthetic_onnx.py --checkpoint /path/to/beauty_regressor.pt \
  --output app/src/main/assets/brow_aesthetic_resnet50.onnx
```

The separate `ml/train_brow_choice_ranker.py` pipeline remains the path to genuine
customer-choice probabilities. It requires consented real sessions with one natural,
one soft, and one defined candidate and a single final choice, separated by customer into
development, calibration, and external test cohorts. The public facial-attractiveness data
cannot substitute for those customer choice labels.
