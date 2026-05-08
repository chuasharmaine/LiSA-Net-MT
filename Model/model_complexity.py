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
            seg_guided_cls=opt["seg_guided_cls"]
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
        "seg_classes": 1,
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
        # MULTITASK
        ("MBDCNN", "multitask"),
        ("BreastCancerMT", "multitask"),
        ("LiSANetMT_NORM", "multitask"),
        ("LiSANetMT_GUIDED", "multitask"),
    ]

    results = []

    for model_name, task in model_list:
        print(f"\nRunning: {model_name} ({task})")
        
        opt = base_opt.copy()
        opt["task"] = task

        if model_name == "LiSANetMT_SEG":
            opt["model_name"] = "LiSANetMT"
            opt["display_name"] = "LiSANetMT (SEG)"
            opt["segmentation"] = True
            opt["classification"] = False
            opt["seg_guided_cls"] = False

        elif model_name == "LiSANetMT_CLS":
            opt["model_name"] = "LiSANetMT"
            opt["display_name"] = "LiSANetMT (CLS)"
            opt["segmentation"] = False
            opt["classification"] = True
            opt["seg_guided_cls"] = False

        elif model_name == "LiSANetMT_NORM":
            opt["model_name"] = "LiSANetMT"
            opt["display_name"] = "LiSANetMT (NORM)"
            opt["segmentation"] = True
            opt["classification"] = True
            opt["seg_guided_cls"] = False

        elif model_name == "LiSANetMT_GUIDED":
            opt["model_name"] = "LiSANetMT"
            opt["display_name"] = "LiSANetMT (GUIDED)"
            opt["segmentation"] = True
            opt["classification"] = True
            opt["seg_guided_cls"] = True

        else:
            opt["model_name"] = model_name
            opt["display_name"] = model_name
            opt["segmentation"] = True
            opt["classification"] = True

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



