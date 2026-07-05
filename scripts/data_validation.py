import os
from PIL import Image


DATA="data/raw/BraTS2019 dataset"



def check_images():

    bad=[]


    for root,dirs,files in os.walk(DATA):

        for file in files:

            path=os.path.join(
                root,
                file
            )

            try:

                img=Image.open(path)
                img.verify()


            except:

                bad.append(path)


    return bad




def class_count():


    for split in ["train","valid","test"]:


        print("\n",split)


        path=os.path.join(
            DATA,
            split
        )


        for cls in os.listdir(path):

            folder=os.path.join(
                path,
                cls
            )


            print(
                cls,
                len(os.listdir(folder))
            )





bad=check_images()


print(
"Corrupted:",
len(bad)
)


class_count()