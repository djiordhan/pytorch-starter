import torchvision


def build_model(num_classes):
    return torchvision.models.segmentation.fcn_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
