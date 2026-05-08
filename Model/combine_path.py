import torch
import argparse

def merge_checkpoints(seg_path, cls_path, output_path):
    # load checkpoints
    seg_ckpt = torch.load(seg_path, map_location="cpu")
    cls_ckpt = torch.load(cls_path, map_location="cpu")

    # try to handle both "raw state_dict" and wrapped dict formats
    def extract_state(ckpt):
        if isinstance(ckpt, dict):
            # common cases
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
            elif "model" in ckpt:
                return ckpt["model"]
        return ckpt

    seg_state = extract_state(seg_ckpt)
    cls_state = extract_state(cls_ckpt)

    # build merged checkpoint
    merged_ckpt = {
        "segmentation_head": seg_state,
        "classification_head": cls_state
    }

    torch.save(merged_ckpt, output_path)

    print(f"\nMerged checkpoint saved to: {output_path}")
    print("Keys:")
    print(merged_ckpt.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seg", required=True, help="path to best segmentation checkpoint")
    parser.add_argument("--cls", required=True, help="path to best classification checkpoint")
    parser.add_argument("--out", default="LiSANetMT_merged.pth", help="output merged checkpoint")

    args = parser.parse_args()

    merge_checkpoints(args.seg, args.cls, args.out)
