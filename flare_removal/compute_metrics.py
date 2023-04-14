import json
import cv2
import numpy as np
from pathlib import Path
from fire import Fire
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def compute_pair_metrics(gt_path: Path, pred_path: Path, bbox = None) -> dict:
    gt_image = cv2.imread(str(gt_path))
    pred_image = cv2.imread(str(pred_path))

    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)

    if bbox is not None:
        gt_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = 0
        pred_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = 0

    gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(pred_image, cv2.COLOR_RGB2GRAY)

    ret = {}

    ret['ssim'] = ssim(gt_gray, pred_gray)
    ret['psnr'] = psnr(gt_image, pred_image)
    return ret


def compute_metrics(gt_path: str, pred_path: str, gn_ann_path: str):
    gt_path = Path(gt_path)
    pred_path = Path(pred_path)

    with open(gn_ann_path, 'r') as f:
        coco = json.load(f)

        images_dict = {coco['images'][i]['id']:coco['images'][i]['file_name'] for i in range(len(coco['images']))}
        bboxes_dict = {images_dict[coco['annotations'][i]['image_id']]:coco['annotations'][i]['bbox'] for i in range(len(coco['annotations']))}
    gt_images = list(gt_path.glob('*'))
    metrics = {}
    for gt_image in tqdm(gt_images):
        pred_image = pred_path / gt_image.name
        assert pred_image.exists(), f'{pred_image} don\'t exists!'

        bbox = bboxes_dict.get(gt_image.name, None)
        ret = compute_pair_metrics(gt_image, pred_image, bbox)
        for k, v in ret.items():
            metrics.setdefault(k, []).append(v)
    metrics = {k: np.mean(v) for k,v in metrics.items()}
    return metrics


if __name__ == "__main__":
    Fire(compute_metrics)
