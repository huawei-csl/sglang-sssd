"""
The EAGLE heads available online are usually provided in float16. This scripts converts the model (inplace)
to bfloat16.
"""

import argparse
import torch
import json

parser = argparse.ArgumentParser(description="Model datatype conversion utility")
parser.add_argument(
    "--model",
    type=str,
    help="Path to the model directory"
)

def convert(model_path: str):
    ckpt = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
    for name, tensor in ckpt.items():
        if tensor.dtype == torch.float16:
            ckpt[name] = tensor.to(torch.bfloat16)
    torch.save(ckpt, f"{model_path}/pytorch_model.bin")

    config_path = f"{model_path}/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["torch_dtype"] = "bfloat16"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Updated torch_dtype to bfloat16.")



if __name__ == "__main__":
    args = parser.parse_args()
    convert(args.model)