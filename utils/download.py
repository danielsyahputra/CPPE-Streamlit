import os

import gdown

def check_dir(dir_name: str) -> bool:
    if os.path.isdir(dir_name):
        return True
    return False

def download_data(dir_name="data") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=1YkJsbUNt-ut1HoSkiQB8rY8Zc5SFzui9", quiet=False
    )
    os.system("tar -xf dataset.tar.gz")
    os.remove("dataset.tar.gz")
    os.chdir("..")

def download_model(dir_name="models") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=18lA8PlHa1z41TCrlNIG318CUQbFsB8No", quiet=False
    )
    gdown.download(
        "https://drive.google.com/uc?id=1wqCbCZob1OTAKfnwnn-hIcTZQUAq2vtQ", quiet=False
    )