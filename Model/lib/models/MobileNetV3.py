import timm

def MobileNetV3(num_classes, pretrained=True):
    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model