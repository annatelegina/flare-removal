import shutil
import secrets
from pathlib import Path
from fire import Fire

def rename_identity(inp_path, out_path, extention='jpg'):
    inp_path = Path(inp_path)
    out_path = Path(out_path)
    out_path.mkdir()
    for img_path in inp_path.glob('*'):
        name = secrets.token_hex(8)
        shutil.copyfile(img_path, out_path/ f"{name}_0_gt.{extention}")
        shutil.copyfile(img_path, out_path/ f"{name}_1_flare.{extention}")

if __name__ == "__main__":
    Fire(rename_identity)