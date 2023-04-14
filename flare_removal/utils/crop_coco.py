import json
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from fire import Fire

 

def get_glimpse(image, bbox):
    bbox = [int(x) for x in bbox]
    glimpse = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
    return glimpse


def crop_coco_annotations(json_path: str, intut_path: str, out_path: str, IMAGE_SHAPE = (3648, 2736)):
    JSON_PATH = json_path
    input_images_path = Path(intut_path)
    out_path = Path(out_path)

    with open(JSON_PATH) as f:
        coco = json.load(f)

    images_filenames = sorted([coco['images'][i]['file_name'] for i in range(len(coco['images']))])
    images_dict = {coco['images'][i]['id']:coco['images'][i]['file_name'] for i in range(len(coco['images']))}
    images_dict_reverted = {coco['images'][i]['file_name']:coco['images'][i]['id'] for i in range(len(coco['images']))}
    bboxes_dict = {coco['annotations'][i]['image_id']:coco['annotations'][i]['bbox'] for i in range(len(coco['annotations']))}

    indexes_sorted = [images_dict_reverted[filename] for filename in images_filenames]
    
    gt_filenames = [images_dict[x].split('_')[0] + '_0_gt.jpg' for x in indexes_sorted]
    
    out_path.mkdir(exist_ok=True)
    for image_id, gt_filename in tqdm(zip(indexes_sorted, gt_filenames), total = len(indexes_sorted)):
        bbox = bboxes_dict[image_id]

        input_filename = images_dict[image_id]
        
        input_image = plt.imread(input_images_path / input_filename)[:,:,:3]
        input_glimpse = get_glimpse(input_image, bbox)
        flare = cv2.copyMakeBorder(input_glimpse,0,IMAGE_SHAPE[0]-input_glimpse.shape[0],0,IMAGE_SHAPE[1]-input_glimpse.shape[1],cv2.BORDER_WRAP)

        gt_image = plt.imread(input_images_path / gt_filename)[:,:,:3]
        gt_glimpse = get_glimpse(gt_image, bbox)
        gt = cv2.copyMakeBorder(gt_glimpse,0,IMAGE_SHAPE[0]-gt_glimpse.shape[0],0,IMAGE_SHAPE[1]-gt_glimpse.shape[1],cv2.BORDER_WRAP)

        name = images_dict[image_id]
        split = name.split('_')
        name_hash = split[0]
        idx = split[1]
        new_name_flare = f'{name_hash}-bbox{idx}_1_flare.jpg'
        new_name_gt = f'{name_hash}-bbox{idx}_0_gt.jpg'
        plt.imsave(str(out_path/new_name_flare), flare)
        plt.imsave(str(out_path/new_name_gt), gt)


    
    


if __name__ == "__main__":
    Fire(crop_coco_annotations)
