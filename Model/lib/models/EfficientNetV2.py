import timm

def EfficientNetV2(num_classes, pretrained=True):
    model = timm.create_model(
        "tf_efficientnetv2_m",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model