import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.pretrained_models import (
    get_mobilenet,
    get_squeezenet,
    get_resnet
)


# -----------------------
# Settings
# -----------------------

DATA_DIR = "data/raw/BraTS2019 dataset"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LR = 0.0001



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# -----------------------
# Transform
# -----------------------

transform = transforms.Compose([


    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),


    transforms.ToTensor(),


    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )

])



train_dataset = datasets.ImageFolder(

    DATA_DIR + "/train",

    transform

)


valid_dataset = datasets.ImageFolder(

    DATA_DIR + "/valid",

    transform

)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE

)



num_classes = len(
    train_dataset.classes
)



# -----------------------
# Training
# -----------------------


def train_model(model,name):


    model.to(device)



    criterion = nn.CrossEntropyLoss()



    optimizer = torch.optim.Adam(

        filter(
            lambda p:p.requires_grad,
            model.parameters()
        ),

        lr=LR

    )



    best_loss=float("inf")



    for epoch in range(EPOCHS):


        model.train()


        train_loss=0



        for images,labels in train_loader:


            images=images.to(device)

            labels=labels.to(device)



            optimizer.zero_grad()



            output=model(images)



            loss=criterion(
                output,
                labels
            )



            loss.backward()

            optimizer.step()



            train_loss += loss.item()




        train_loss /= len(train_loader)



        # validation


        model.eval()


        val_loss=0



        with torch.no_grad():


            for images,labels in valid_loader:


                images=images.to(device)

                labels=labels.to(device)



                output=model(images)



                loss=criterion(
                    output,
                    labels
                )


                val_loss += loss.item()



        val_loss /= len(valid_loader)



        print(

            f"{name} Epoch {epoch+1}/{EPOCHS} "
            f"Train {train_loss:.4f} "
            f"Val {val_loss:.4f}"

        )



        if val_loss < best_loss:


            best_loss=val_loss



            os.makedirs(

                "results/pretrained_models",

                exist_ok=True

            )


            torch.save(

                model.state_dict(),

                f"results/pretrained_models/{name}.pth"

            )



# -----------------------
# Run Models
# -----------------------


models = {


"MobileNetV2":
get_mobilenet(num_classes),



"SqueezeNet":
get_squeezenet(num_classes),



"ResNet18":
get_resnet(num_classes)

}



for name,model in models.items():


    train_model(
        model,
        name
    )



print("Pretrained training completed")