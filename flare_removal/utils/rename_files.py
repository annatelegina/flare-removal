from pathlib import Path
from fire import Fire


def rename_files(data_dir: str):
    """
        It is used for file renaming to their positional index (example: <first> -> 001)
        Useful for mapping file cration for align_images.py script
    """
    data_dir = Path(data_dir)

    files = list(sorted(data_dir.glob('*')))
    n_files = len(files)
    width = len(str(n_files))
    for i, f in enumerate(files):
        parent = f.parents[0]
        extension = f.suffix
        new_name = str(i).zfill(width) + extension
        f.rename(parent / new_name)


if __name__ == "__main__":
    Fire(rename_files)
