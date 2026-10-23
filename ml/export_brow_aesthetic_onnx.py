#!/usr/bin/env python3
"""Export the SCUT-FBP5500 ResNet-50 human-rating regressor to ONNX for local Android inference.

Research/education only. This model predicts an averaged whole-face rating in [1, 5].
It does NOT predict individual customer choice or objective eyebrow suitability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 10_000_000:
        raise SystemExit(f"Missing or unexpectedly small SCUT checkpoint: {args.checkpoint}")

    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            str(args.output),
            input_names=["image"],
            output_names=["rating"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(onnx.load(str(args.output)))
    runtime = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    meta = runtime.get_inputs()[0]
    if meta.shape != [1, 3, 224, 224]:
        raise SystemExit(f"Unexpected ONNX input shape: {meta.shape}")
    output = runtime.run(None, {meta.name: np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    if output.shape not in ((1, 1), (1,)) or not np.isfinite(output).all():
        raise SystemExit(f"Invalid ONNX output: shape={output.shape}, values={output}")
    if args.output.stat().st_size < 10_000_000:
        raise SystemExit("ONNX export is unexpectedly small.")
    print(f"SCUT aesthetic regressor ONNX OK: {args.output} ({args.output.stat().st_size} bytes), output={output.tolist()}")


if __name__ == "__main__":
    main()
