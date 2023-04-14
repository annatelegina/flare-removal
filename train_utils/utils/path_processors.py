from functools import partial
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from guided_filter_tf.guided_filter import guided_filter

ShapeType = Tuple[int, int]


class BasePathProcessor:
    OUT_SPEC = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )

    def __init__(self,
                 resize_shape: Optional[ShapeType],
                 transforms,
                 min_alpha: float = 1,
                 r: int = 16):
        if resize_shape is not None:
            self._resize = partial(tf.image.resize, size=resize_shape, method='bilinear',
                                   preserve_aspect_ratio=False, antialias=True)
        else:
            self._resize = lambda x: x
    
        self._min_alpha = min_alpha
        self._r = r
        self._transform = transforms

    def _load_img(self, path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        return img
        
    def apply_aug(self, img, img_gt):
        transformed = self._transform(image=img, image0=img_gt)
        img = transformed['image']
        img_gt = transformed['image0']
        img = tf.cast(img, tf.float32)
        img_gt = tf.cast(img_gt, tf.float32)
        return img, img_gt

    def _preprocess_img(self, img_flare, img_gt):
        img_flare, img_gt = self.apply_aug(img_flare, img_gt)

        return img_flare, img_gt

    def _guided_filter(self, img_flare, img_gt):
        if self._r is not None:
            img_gt = guided_filter(x=img_gt[tf.newaxis, ...], y=img_flare[tf.newaxis, ...], r=self._r, nhwc=True)
            img_gt = tf.clip_by_value(img_gt, clip_value_min=0, clip_value_max=1)

        return img_flare, img_gt[0]

    @tf.function
    def load(self, gt_file_path, flare_path):
        img_flare = self._load_img(flare_path) / 255
        img_gt = self._load_img(gt_file_path) / 255

        img_flare = self._resize(img_flare)
        img_gt = self._resize(img_gt)

        img_flare, img_gt = self._guided_filter(img_flare, img_gt)
        img_flare = tf.cast(img_flare * 255, tf.uint8)
        img_gt = tf.cast(img_gt * 255, tf.uint8)

        return img_flare, img_gt

    def aug(self, imgs):
        imgs = tuple(map(lambda img: (img / 255).astype(np.float32), imgs))
        img_flare, img_gt = self._preprocess_img(*imgs)

        if self._min_alpha is not None:
            alpha = tf.random.uniform([1], minval=self._min_alpha, maxval=1, dtype=tf.float32)
            alpha = alpha[0]
        else:
            alpha = 1

        img_gt = alpha * img_gt + (1 - alpha) * img_flare[..., 0:3]

        img_gt = tf.clip_by_value(img_gt, clip_value_min=0, clip_value_max=1)
        img_flare = tf.clip_by_value(img_flare, clip_value_min=0, clip_value_max=1)
        
        return img_flare, img_gt, alpha

    def __call__(self, gt_file_path, flare_file_path):
        imgs = self.load(gt_file_path, flare_file_path)
        return self.aug(imgs)


class FullresPathProcessor(BasePathProcessor):
    OUT_SPEC = (tf.TensorSpec(shape=(None, None, None, 6), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )
    def __init__(self,
                 transforms,
                 guidance_ds: int = 8,
                 resize_shape: Optional[ShapeType] = None,
                 min_alpha: float = 1,
                 r: int = 16):
        super().__init__(resize_shape=resize_shape,
                 transforms=transforms,
                 min_alpha=min_alpha,
                 r=r)
        self._resize_shape = resize_shape
        self._guidance_ds = guidance_ds

    def apply_aug(self, img, img_guidance, img_gt):
        transformed = self._transform(image=img, image0=img_guidance, image1=img_gt)
        img = transformed['image']
        img_guidance = transformed['image0']
        img_gt = transformed['image1']
        img = tf.cast(img, tf.float32)
        img_guidance = tf.cast(img_guidance, tf.float32)
        img_gt = tf.cast(img_gt, tf.float32)
        return img, img_guidance, img_gt

    def _get_downscaled_gt(self, img_gt):
        shape = tf.shape(img_gt)
        img_guide = tf.image.resize(img_gt, shape[:2] // self._guidance_ds, antialias=True)

        return img_guide

    @tf.function
    def load(self, gt_file_path, flare_path):
        img_flare = self._load_img(flare_path) / 255
        img_gt = self._load_img(gt_file_path) / 255
        img_flare, img_gt = self._guided_filter(img_flare, img_gt)
        img_guide = self._get_downscaled_gt(img_gt)

        img_flare = self._resize(img_flare)
        img_gt = self._resize(img_gt)
        img_guide = tf.image.resize(img_guide, tf.shape(img_gt)[:2], antialias=True)

        img_flare = tf.cast(img_flare * 255, tf.uint8)
        img_gt = tf.cast(img_gt * 255, tf.uint8)
        img_guide = tf.cast(img_guide * 255, tf.uint8)

        return img_flare, img_gt, img_guide

    def _preprocess_img(self, img_flare, img_gt, img_guide):                
        img_flare, img_guide, img_gt = self.apply_aug(img_flare, img_guide, img_gt)
        img_flare = tf.concat([img_flare, img_guide], axis=-1)

        return img_flare, img_gt