import torch
import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from models.custom_cnn import CNN3, CNN4, CNN5



DATA_DIR="data/raw/BraTS2019 dataset/test"


MODEL_PATH="results/custom_models/CNN5.pth"


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



model=CNN5(
    num_classes=2
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
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


        outputs=model(images)


        predictions=torch.argmax(
            outputs,
            dim=1
        )


        y_true.extend(
            labels.numpy()
        )


        y_pred.extend(
            predictions.cpu().numpy()
        )



print(
classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
)
)



cm=confusion_matrix(
    y_true,
    y_pred
)



print("Confusion Matrix")

print(cm)



disp=ConfusionMatrixDisplay(
    cm,
    display_labels=test_dataset.classes
)


disp.plot()


os.makedirs(
    "results/plots",
    exist_ok=True
)


plt.savefig(
    "results/plots/confusion_matrix.png"
)


plt.show()