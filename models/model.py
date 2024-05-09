from typing import Callable

from models.mobile_vit import MobileVIT
from models.mobilenet import MobileNetV2
from models.resnet import ResNet20, ResNet32, ResNet44, ResNet56


def get_model(
    model_type: str, num_classes: int, quantize_fn: Callable, bits: int, image_size
):
    if model_type == "resnet20":
        model = ResNet20(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet32":
        model = ResNet32(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet44":
        model = ResNet44(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet56":
        model = ResNet56(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "mobilenet":
        model = MobileNetV2(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "mobilevit":
        model = MobileVIT(
            num_classes=num_classes,
            quantize_fn=quantize_fn,
            bits=bits,
            image_size=image_size,
        )

    return model
