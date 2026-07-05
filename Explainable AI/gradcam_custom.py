import torch
import cv2
import os

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.custom_cnn import CNN5



IMAGE_PATH="data/raw/BraTS2019 dataset/test/yes/Y1.jpg"


MODEL_PATH="results/custom_models/CNN5.pth"



device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



transform=transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

])



image=Image.open(
    IMAGE_PATH
).convert("RGB")



input_tensor=transform(
    image
).unsqueeze(0).to(device)



model=CNN5(2)


model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )

)


model.to(device)

model.eval()



target_layers=[

    model.model[12]

]



cam=GradCAM(

    model=model,

    target_layers=target_layers

)



grayscale_cam=cam(
    input_tensor=input_tensor
)[0]



rgb_image=input_tensor.cpu()[0].permute(1,2,0).numpy()



visualization=show_cam_on_image(

    rgb_image,

    grayscale_cam,

    use_rgb=True

)



os.makedirs(
    "results/xai",
    exist_ok=True
)



cv2.imwrite(

    "results/xai/custom_gradcam.jpg",

    visualization

)



print(
"Custom CNN GradCAM saved"
)