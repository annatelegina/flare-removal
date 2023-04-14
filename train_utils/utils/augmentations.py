import random
import cv2
import numpy as np
from pathlib import Path
from albumentations.core.transforms_interface import ImageOnlyTransform, BasicTransform

class FlareAdder:
    def __init__(self, path, transforms):
        path = Path(path)
        gt = list(path.glob("*_gt*"))
        assert len(gt) == 1
        
        flares = list(path.glob("*_flare*"))
        assert len(flares) > 0
        
        self.gt = self._load_img(gt[0])
        self.flares = flares
        self.transforms = transforms
        
    def aug_flare(self, flare, gt):
        transformed = self.transforms(image=flare, image0=gt)
        flare = transformed['image']
        gt = transformed['image0']
        
        return flare, gt
    
    def fuse_flare(self, image, flare):
        fused = cv2.addWeighted(image, 1, flare, 1, 0)
        return fused
    
    def _load_img(self, path):
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_flare(self, min_size, image_shape):
        flare = random.choice(self.flares)
        flare = self._load_img(flare)
        flare, gt = self.aug_flare(flare, self.gt)

        h = random.randint(min_size, image_shape[0])
        w = random.randint(min_size, image_shape[1])
        
        flare_res, gt_res = self._resize_flare(flare, gt, image_shape, h, w)

        flare_res = flare_res / 255
        flare_res = flare_res * flare_res * flare_res

        gt_res = gt_res / 255
        return flare_res.astype(np.float32), gt_res.astype(np.float32)

    def _resize_flare(self, flare, gt, image_shape, h, w):
        flare_res = cv2.resize(flare, (w, h))
        gt_res = cv2.resize(gt, (w, h))
        
        top_p = random.randint(0, image_shape[0] - flare_res.shape[0])
        bot_p = image_shape[0] - flare_res.shape[0] - top_p
        left_p = random.randint(0, image_shape[1] - flare_res.shape[1])
        right_p = image_shape[1] - flare_res.shape[1] - left_p
        
        flare_res = np.pad(flare_res, [(top_p, bot_p), (left_p, right_p), (0,0)])
        gt_res = np.pad(gt_res, [(top_p, bot_p), (left_p, right_p), (0,0)])
        return flare_res, gt_res

class FlareAug(BasicTransform):
    def __init__(self, flare_adder: FlareAdder, min_size=512, always_apply=False, p=0.5):
        self._flare_adder = flare_adder
        self._min_size = min_size
        super().__init__(always_apply, p)
    
    @property
    def targets(self):
        return {
            "image": self.apply_flare,
            "image0": self.apply_gt,
        }

    @property
    def targets_as_params(self):
        return ["image"]

    def apply_flare(self, img, flare, **params):
        return self._flare_adder.fuse_flare(img, flare)
        
    def apply_gt(self, img, gt, **params):
        return self._flare_adder.fuse_flare(img, gt)

    def get_params_dependent_on_targets(self, params):
        flare, gt = self._flare_adder.load_flare(self._min_size, params["image"].shape)
        
        return {
            "gt": gt,
            "flare": flare
        }


class ZeroChannelAug(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.25):
        super().__init__(always_apply, p)
    
    def apply(self, img, channel, ch_proba, ch_mult,  **params):
        img = img.copy()
        for ch in range(3):
                if ch != channel:
                    if ch_proba > 0.5:
                        img[..., ch] = np.zeros_like(img[..., ch])
                    else:
                        img[..., ch] = img[..., ch] * ch_mult[ch]
        return img

    def get_params(self):
        
        return {
            "channel": np.random.randint(0, 3),
            "ch_proba": np.random.uniform(),
            "ch_mult": np.random.uniform(size=3)
        }


class InputFusionAug(BasicTransform):
    def __init__(self, always_apply=False, p=0.25):
        super().__init__(always_apply, p)

    @property
    def targets(self):
        return {
            "image": self.apply_to_flare_image,
            "image_0": self.apply_to_gt_image,
        }

    @property
    def targets_as_params(self):
        return ["image_0"]

    def apply_to_flare_image(self, image, gt_image, alpha, **params):
        flare_image = np.clip(cv2.addWeighted(image, (1 - alpha), gt_image, alpha, 0), 0, 255)
        return flare_image

    def apply_to_gt_image(self, image_0, **params):
        return image_0

    def get_params(self):
        return {
            "alpha": np.random.uniform(low=0, high=0.5)
        }

    def get_params_dependent_on_targets(self, params):
        gt = params["image_0"]

        return {
            "gt_image": gt
        }
