import platform
import os
import git
from pathlib import Path

HOSTNAME = platform.uname()[1]
EMAIL = git.Repo(search_parent_directories=True).config_reader().get_value("user", "email")

if HOSTNAME == 'DESKTOP-ISTVT9J':
    dataset_dir = Path(r"H:\Flare_removal_datasets")
    log_dir = Path(r"G:\logs")
elif HOSTNAME == 'gpuserver1':
    dataset_dir = Path(r"/shared/data1/Flare_removal_datasets")
    log_dir = Path(r"/shared/data2/flare_removal_logs")
elif HOSTNAME == 'gpuserver184':
    dataset_dir = Path(r"/shared/data2/flare_removal_datasets")
    log_dir = Path(r"/shared/data2/flare_removal_logs")
else:
    raise ValueError("Unknown host")

LOG_DIR = log_dir / EMAIL.split("@")[0]
LOG_DIR.mkdir(exist_ok=True)

TRAIN_FOLDERS = [
                 r"20220517_fili",
                 r"20220517_fili_cropped",
                 r"20220518_sml_day",
                 r"20220518_sml_day_cropped",
                 r"20220519_kutuzovskiy",
                 r"20220519_kutuzovskiy_cropped",
                 r"20220520_flare_removal_daylight_sml_new",
                 r"20220520_flare_removal_daylight_sml_new_cropped",
                 r"20220531_identity",
                 r"20220602_arbat",
                 r"20220602_identity_arbat",
                 ]

TRAIN_FOLDERS = [dataset_dir / folder for folder in TRAIN_FOLDERS]