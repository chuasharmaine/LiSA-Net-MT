import torch
from thop import profile

from lib.models import UNet
from lib.models import PMFSNet
from lib.models import EGEUNet
from lib.models import LiSANet

from lib.models import ResNet50
from lib.models import DenseNet121
from lib.models import EfficientNetV2
from lib.models import MobileNetV3

from lib.models import MBDCNN
from lib.models import BreastCancerMT
from lib.models import LiSANetMT


def build_model(opt):
    name = opt["model_name"]

    # SEGMENTATION
    if name == "UNet":
        return UNet(
            n_channels=opt["in_channels"],
            n_classes=opt["seg_classes"]
        )
    
    elif name == "PMFSNet":
        return PMFSNet(
            in_channels=opt["in_channels"],
            out_channels=opt["seg_classes"],
            dim=opt["dimension"],
            scaling_version=opt["scaling_version"]
        )

    elif name == "EGEUNet":
        return EGEUNet(
            input_channels=opt["in_channels"],
            num_classes=opt["seg_classes"]
        )

    elif name == "LiSANet":
        return LiSANet(
            in_channels=opt["in_channels"],
            out_channels=opt["seg_classes"],
            dim=opt["dimension"],
            scaling_version=opt["scaling_version"]
        )


    # CLASSIFICATION
    elif name == "ResNet50":
        return ResNet50(
            num_classes=opt["cls_classes"],
            pretrained=False
        )

    elif name == "DenseNet121":
        return DenseNet121(
            num_classes=opt["cls_classes"],
            pretrained=False
        )

    elif name == "EfficientNetV2":
        return EfficientNetV2(
            num_classes=opt["cls_classes"],
            pretrained=False
        )

    elif name == "MobileNetV3":
        return MobileNetV3(
            num_classes=opt["cls_classes"],
            pretrained=False
        )

    # MULTITASK
    elif name == "LiSANetMT":
        return LiSANetMT(
            in_channels=opt["in_channels"],
            seg_out_channels=opt["seg_classes"],
            cls_out_channels=opt["cls_classes"],
            dim=opt["dimension"],
            scaling_version=opt["scaling_version"],
            segmentation=opt["segmentation"],
            classification=opt["classification"],
            seg_guided_cls=opt["seg_guided_cls"],
            cls_head_variant=opt.get("cls_head_variant", "baseline")
        )

    elif name == "MBDCNN":
        return MBDCNN(
            in_channels=opt["in_channels"],
            seg_out_channels=opt["seg_classes"],
            cls_out_channels=opt["cls_classes"]
        )

    elif name == "BreastCancerMT":
        return BreastCancerMT(
            in_channels=opt["in_channels"],
            seg_out_channels=opt["seg_classes"],
            cls_out_channels=opt["cls_classes"]
        )

    else:
        raise ValueError(f"Unknown model: {name}")


# COMPLEXITY CALCULATOR
def get_model_complexity(opt):

    model = build_model(opt)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dummy input
    input_tensor = torch.randn(
        1,
        opt["in_channels"],
        opt["img_size"],
        opt["img_size"]
    ).to(device)

    # FLOPs + Params
    flops, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )

    print(f"Model : {opt['display_name']}")
    print(f"Params: {params / 1e6:.4f} M")
    print(f"FLOPs : {flops / 1e9:.4f} GFLOPs")

    return flops, params

# MAIN
if __name__ == "__main__":

    base_opt = {
        "in_channels": 3,
        # Match train.py multitask/segmentation configuration.
        "seg_classes": 2,
        "cls_classes": 2,
        "img_size": 224,
        "dimension": "2d",
        "scaling_version": "BASIC"
    }

    # MODEL LIST
    model_list = [
        # SEGMENTATION
        ("UNet", "segmentation"),
        ("PMFSNet", "segmentation"),
        ("EGEUNet", "segmentation"),
        ("LiSANet", "segmentation"),
        ("LiSANetMT_SEG", "segmentation"),
        # CLASSIFICATION
        ("ResNet50", "classification"),
        ("DenseNet121", "classification"),
        ("EfficientNetV2", "classification"),
        ("MobileNetV3", "classification"),
        ("LiSANetMT_CLS", "classification"),
        ("LiSANetMT_CLS_LESION_FUSION", "classification"),
        ("LiSANetMT_CLS_LARGER_MLP", "classification"),
        ("LiSANetMT_CLS_AVGMAX", "classification"),
        ("LiSANetMT_CLS_MULTISCALE", "classification"),
        ("LiSANetMT_CLS_MULTISCALE_LARGER", "classification"),
        ("LiSANetMT_CLS_PROJECTED_FUSION", "classification"),
        ("LiSANetMT_CLS_PROJECTED_FUSION_SE", "classification"),
        # MULTITASK
        ("MBDCNN", "multitask"),
        ("BreastCancerMT", "multitask"),
        ("LiSANetMT_NORM", "multitask"),
        ("LiSANetMT_GUIDED", "multitask"),
        ("LiSANetMT_LESION_FUSION", "multitask"),
        ("LiSANetMT_LESION_FUSION_GUIDED", "multitask"),
    ]

    results = []

    for model_name, task in model_list:
        print(f"\nRunning: {model_name} ({task})")
        
        opt = base_opt.copy()
        opt["task"] = task

        lisanet_configs = {
            "LiSANetMT_SEG": ("LiSANetMT (SEG)", True, False, False, "baseline"),
            "LiSANetMT_CLS": ("LiSANetMT (CLS Base)", False, True, False, "baseline"),
            "LiSANetMT_CLS_LESION_FUSION": ("CLS Lesion Fusion", False, True, False, "lesion_fusion"),
            "LiSANetMT_CLS_LARGER_MLP": ("CLS Larger MLP", False, True, False, "larger_mlp"),
            "LiSANetMT_CLS_AVGMAX": ("CLS AVG MAX", False, True, False, "avgmax"),
            "LiSANetMT_CLS_MULTISCALE": ("CLS Multiscale", False, True, False, "multiscale"),
            "LiSANetMT_CLS_MULTISCALE_LARGER": ("CLS Multiscale + Larger", False, True, False, "multiscale_larger"),
            "LiSANetMT_CLS_PROJECTED_FUSION": ("CLS Projected Fusion", False, True, False, "projected_fusion"),
            "LiSANetMT_CLS_PROJECTED_FUSION_SE": ("CLS Projected Fusion SE", False, True, False, "projected_fusion_se"),
            "LiSANetMT_NORM": ("LiSANetMT", True, True, False, "baseline"),
            "LiSANetMT_GUIDED": ("LiSANetMT Seg-Guided", True, True, True, "baseline"),
            "LiSANetMT_LESION_FUSION": ("LiSANetMT Lesion Fusion", True, True, False, "lesion_fusion"),
            "LiSANetMT_LESION_FUSION_GUIDED": ("LiSANetMT LF Seg-Guided", True, True, True, "lesion_fusion"),
        }

        if model_name in lisanet_configs:
            display_name, segmentation, classification, guided, variant = lisanet_configs[model_name]
            opt["model_name"] = "LiSANetMT"
            opt["display_name"] = display_name
            opt["segmentation"] = segmentation
            opt["classification"] = classification
            opt["seg_guided_cls"] = guided
            opt["cls_head_variant"] = variant

        else:
            opt["model_name"] = model_name
            opt["display_name"] = model_name
            opt["segmentation"] = True
            opt["classification"] = True
            opt["seg_guided_cls"] = False
            opt["cls_head_variant"] = "baseline"

        try:
            flops, params = get_model_complexity(opt)
            results.append({
                "model": opt["display_name"],
                "task": task,
                "params(M)": params / 1e6,
                "flops(G)": flops / 1e9
            })


        except Exception as e:
            print(f"Failed: {model_name} → {e}")

    print("\n\nFINAL")

    for r in results:
        print(
            f"{r['model']:20} | "
            f"{r['task']:14} | "
            f"{r['params(M)']:.2f} M | "
            f"{r['flops(G)']:.2f} GFLOPs"
        )