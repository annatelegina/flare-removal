import secrets
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from fire import Fire
from tqdm import tqdm 

sys.path.append(str(Path(__file__).absolute().parents[2]))
from train_utils.utils import image_transform

def align_images(data_path: str, out_path: str, mapping_file: str, extention: str = 'jpg'):
    """
        mapping_file: file that contains lines "<target id> <sources start id> <sources end id>"
    """

    data_path = Path(data_path)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    files = list(sorted(data_path.glob(f'*.{extention}')))


    with open(mapping_file, 'r') as f:
        for line in tqdm(f):
            if len(line.strip()) == 0: # Skip empty line
                continue
            target_id, source_start, source_end = line.split()
            source_ids = list(range(int(source_start), int(source_end)+1))

            target_path = files[int(target_id)]
            sources = [files[int(x)] for x in source_ids]

            name = secrets.token_hex(8)

            target_image = plt.imread(target_path)
            plt.imsave(out_path / (name + f'.{extention}'), target_image)
            for i, source_path in enumerate(sources):
                if source_path == target_path:
                    continue
                source_image = plt.imread(source_path)
                aligned_image = image_transform.align_images(target_image, source_image)
                plt.imsave(out_path / (name + f'_{i}.{extention}'), aligned_image)

if __name__ == "__main__":
    Fire(align_images)
