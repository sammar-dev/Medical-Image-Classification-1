import os


DATA_PATH="data/raw/BraTS2019_dataset"


if os.path.exists(DATA_PATH):

    print("Dataset already exists")

else:

    print(
        "Please manually place dataset in data/raw/"
    )