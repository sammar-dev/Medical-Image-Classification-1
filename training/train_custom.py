import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.custom_cnn import CNN3, CNN4, CNN5


# -------------------------
# Settings
# -------------------------

DATA_DIR = "data/raw/BraTS2019 dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -------------------------
# Data preparation
# -------------------------

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


train_data = datasets.ImageFolder(
    DATA_DIR + "/train",
    transform
)


valid_data = datasets.ImageFolder(
    DATA_DIR + "/valid",
    transform
)



train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)



num_classes = len(train_data.classes)



# -------------------------
# Training function
# -------------------------


def train_model(model,name):


    model.to(DEVICE)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    train_losses=[]
    val_losses=[]


    best_loss=float("inf")


    for epoch in range(EPOCHS):


        model.train()

        total_loss=0



        for images,labels in train_loader:


            images=images.to(DEVICE)

            labels=labels.to(DEVICE)


            optimizer.zero_grad()


            output=model(images)


            loss=criterion(
                output,
                labels
            )


            loss.backward()

            optimizer.step()



            total_loss += loss.item()



        train_loss = total_loss / len(train_loader)


        # validation

        model.eval()

        val_loss=0


        with torch.no_grad():


            for images,labels in valid_loader:


                images=images.to(DEVICE)

                labels=labels.to(DEVICE)


                output=model(images)


                loss=criterion(
                    output,
                    labels
                )


                val_loss += loss.item()



        val_loss /= len(valid_loader)


        train_losses.append(train_loss)

        val_losses.append(val_loss)



        print(
            f"{name} Epoch {epoch+1}/{EPOCHS} "
            f"Train:{train_loss:.4f} "
            f"Val:{val_loss:.4f}"
        )



        if val_loss < best_loss:


            best_loss=val_loss


            os.makedirs(
                "results/custom_models",
                exist_ok=True
            )


            torch.save(
                model.state_dict(),
                f"results/custom_models/{name}.pth"
            )


    # Learning curve

    plt.figure(figsize=(7,5))


    plt.plot(
        train_losses,
        label="Train Loss"
    )


    plt.plot(
        val_losses,
        label="Validation Loss"
    )


    plt.title(name)


    plt.xlabel("Epoch")

    plt.ylabel("Loss")


    plt.legend()


    os.makedirs(
        "results/plots",
        exist_ok=True
    )


    plt.savefig(
        f"results/plots/{name}_curve.png"
    )


    plt.close()



# -------------------------
# Run models
# -------------------------


models = {

"CNN3": CNN3(num_classes),

"CNN4": CNN4(num_classes),

"CNN5": CNN5(num_classes)

}



for name,model in models.items():

    train_model(
        model,
        name
    )


print("Training completed")
