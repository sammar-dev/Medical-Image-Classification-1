import torch
import torch.nn as nn


# CNN 3 Layers
class CNN3(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(128*28*28,512),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,num_classes)
        )


    def forward(self,x):

        return self.model(x)



# CNN 4 Layers
class CNN4(nn.Module):

    def __init__(self,num_classes=2):

        super().__init__()

        self.model=nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Flatten(),

            nn.Linear(256*14*14,512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,num_classes)

        )


    def forward(self,x):

        return self.model(x)




# CNN 5 Layers
class CNN5(nn.Module):

    def __init__(self,num_classes=2):

        super().__init__()


        self.model=nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Flatten(),


            nn.Linear(512*7*7,512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,num_classes)

        )


    def forward(self,x):

        return self.model(x)