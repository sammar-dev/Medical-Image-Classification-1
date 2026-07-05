import os

from torchvision import transforms,datasets
from torch.utils.data import DataLoader



DATA="data/raw/BraTS2019 dataset"


transform=transforms.Compose([


transforms.Resize(
(224,224)
),


transforms.ToTensor(),


transforms.Normalize(

[0.485,0.456,0.406],

[0.229,0.224,0.225]

)

])




train_dataset=datasets.ImageFolder(

DATA+"/train",

transform

)


valid_dataset=datasets.ImageFolder(

DATA+"/valid",

transform

)


test_dataset=datasets.ImageFolder(

DATA+"/test",

transform

)




train_loader=DataLoader(

train_dataset,

batch_size=32,

shuffle=True

)


valid_loader=DataLoader(

valid_dataset,

batch_size=32

)


test_loader=DataLoader(

test_dataset,

batch_size=32

)



print(
train_dataset.classes
)


print(
"Preprocessing completed"
)