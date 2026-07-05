import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    squeezenet1_1,
    resnet18,
    MobileNet_V2_Weights,
    SqueezeNet1_1_Weights,
    ResNet18_Weights
)



def get_mobilenet(num_classes=2):

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )


    for param in model.features.parameters():
        param.requires_grad=False


    model.classifier[1]=nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )


    return model




def get_squeezenet(num_classes=2):


    model=squeezenet1_1(
        weights=SqueezeNet1_1_Weights.DEFAULT
    )


    for param in model.features.parameters():
        param.requires_grad=False



    model.classifier[1]=nn.Conv2d(
        512,
        num_classes,
        kernel_size=1
    )


    model.num_classes=num_classes


    return model





def get_resnet(num_classes=2):


    model=resnet18(
        weights=ResNet18_Weights.DEFAULT
    )


    for param in model.parameters():
        param.requires_grad=False



    model.fc=nn.Linear(
        model.fc.in_features,
        num_classes
    )


    return model