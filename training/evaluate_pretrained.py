import torch
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from models.pretrained_models import (
    get_mobilenet,
    get_squeezenet,
    get_resnet
)


DATA_DIR="data/raw/BraTS2019 dataset/test"


IMAGE_SIZE=224

BATCH_SIZE=32



device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



transform=transforms.Compose([


    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),


    transforms.ToTensor(),


    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])



test_dataset=datasets.ImageFolder(

    DATA_DIR,

    transform

)



test_loader=DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)



models={


"MobileNetV2":

get_mobilenet(2),



"SqueezeNet":

get_squeezenet(2),



"ResNet18":

get_resnet(2)

}




paths={


"MobileNetV2":

"results/pretrained_models/MobileNetV2.pth",


"SqueezeNet":

"results/pretrained_models/SqueezeNet.pth",


"ResNet18":

"results/pretrained_models/ResNet18.pth"

}



for name,model in models.items():


    print("\n================")

    print(name)


    model.load_state_dict(

        torch.load(

            paths[name],

            map_location=device

        )

    )


    model.to(device)

    model.eval()



    y_true=[]

    y_pred=[]



    with torch.no_grad():


        for images,labels in test_loader:


            images=images.to(device)



            output=model(images)



            pred=torch.argmax(
                output,
                dim=1
            )


            y_true.extend(
                labels.numpy()
            )


            y_pred.extend(
                pred.cpu().numpy()
            )



    print(

        "Accuracy:",

        accuracy_score(
            y_true,
            y_pred
        )

    )


    print(

        classification_report(

            y_true,

            y_pred,

            target_names=test_dataset.classes

        )

    )


    print(

        "Confusion Matrix"

    )


    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )