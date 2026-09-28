"""
Utility to export trained Sfiral ternary models to ONNX format.
Bakes ternary weights into standard FP Linear nodes for smooth inference.
"""

from __future__ import annotations
import argparse
import os
import sys

# Configure UTF-8 encoding for Windows console to prevent PyTorch's unicode checkmarks from crashing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from quant.ternarize import TernaryLinear, ternarize_weight, TernaryConfig
from training.train import SfiralTernaryNet

def convert_module_to_onnx_ready(m: nn.Module) -> nn.Module:
    if isinstance(m, TernaryLinear):
        cfg = m.quant.cfg if hasattr(m, "quant") else TernaryConfig()
        with torch.no_grad():
            w_q, _, _ = ternarize_weight(m.weight, cfg)
        lin = nn.Linear(m.in_features, m.out_features, bias=(m.bias is not None))
        lin.weight.data.copy_(w_q)
        if m.bias is not None:
            lin.bias.data.copy_(m.bias.data)
        return lin

    for name, child in list(m.named_children()):
        setattr(m, name, convert_module_to_onnx_ready(child))
    return m

def export_to_onnx(ckpt_path: str, onnx_path: str):
    print(f"[+] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model = SfiralTernaryNet()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Запекаем тернарные веса в обычные слои для ONNX
    model = convert_module_to_onnx_ready(model)
    model.eval()

    dummy_input = torch.randn(1, 8, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["sfiral_node_params"],
        output_names=["topology_logits"],
        dynamic_axes={"sfiral_node_params": {0: "batch_size"}, "topology_logits": {0: "batch_size"}}
    )
    print(f"[OK] Model successfully exported to ONNX -> {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Export Sfiral Ternary Model to ONNX")
    parser.add_argument("--ckpt", type=str, default="./models/sfiral_ternary_copilot.pt")
    parser.add_argument("--onnx", type=str, default="./models/sfiral_ternary_copilot.onnx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.onnx) or ".", exist_ok=True)
    export_to_onnx(args.ckpt, args.onnx)

if __name__ == "__main__":
    main()