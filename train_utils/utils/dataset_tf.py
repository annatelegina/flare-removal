from pathlib import Path
from typing import Union, Tuple, List

import tensorflow as tf
from guided_filter_tf.guided_filter import guided_filter


ShapeType = Tuple[int, int]


def get_dataset(data_paths: List[Union[str, Path]]):
    data_paths = [Path(x) for x in data_paths]

    gt_files = sum((list(x.glob('*_gt.*')) for x in data_paths), [])
    
    sorted_gt = sorted(gt_files)
    flare_files = [len(list(x.parents[0].glob(x.name.split('_')[0] + '_*'))) for x in sorted_gt]
    extended_gt = sum([[pair[0]]*pair[1] for pair in zip(sorted_gt, flare_files)],[])
    flare_paths = sum([sorted(list(x.parents[0].glob(x.name.split('_')[0] + '_*')))[:] for x in sorted_gt], [])
    
    #flare_files = [len(list(x.parents[0].glob(x.name.split('_')[0] + '*'))) - 1 for x in gt_files]
    #gt_files = list(map(str, gt_files))
    
    gt_files = list(map(str, extended_gt))
    flare_files = list(map(str, flare_paths))

    for gt, flare in zip(gt_files, flare_files):
        assert gt.split('_')[0] == flare.split('_')[0]

    gt_ds = tf.data.Dataset.from_tensor_slices(gt_files)
    flare_df = tf.data.Dataset.from_tensor_slices(flare_files)
    list_ds = tf.data.Dataset.zip((gt_ds, flare_df))
    return list_ds


def pad_image_to_match_model_downscaling(image, model_downscale_ratio):
    image_shape = tf.shape(image)
    height, width = image_shape[1], image_shape[2]
    pad_0 = (model_downscale_ratio - height % model_downscale_ratio) % model_downscale_ratio
    pad_1 = (model_downscale_ratio - width % model_downscale_ratio) % model_downscale_ratio
    padded_image = tf.pad(image, [[0, 0], [0, pad_0], [0, pad_1], [0, 0]], "SYMMETRIC")
    return padded_image


@tf.function
def get_fullres_guidance(flare_image, scale_ratio, model_downscale_ratio, downscale_model):
    flare_image = tf.expand_dims(flare_image, axis=0)
    flare_image = tf.ensure_shape(flare_image, [None, None, None, 3])
    flare_image_shape = tf.shape(flare_image)
    height_fullres, width_fullres = flare_image_shape[1], flare_image_shape[2]
    flare_image = tf.image.resize(flare_image, size=[height_fullres // scale_ratio, width_fullres // scale_ratio],
                                  method='bilinear',
                                  preserve_aspect_ratio=False,
                                  antialias=True)
    padded_image = pad_image_to_match_model_downscaling(flare_image, model_downscale_ratio)
    cnn_out = downscale_model(padded_image)
    cnn_out = cnn_out[:, :height_fullres // scale_ratio, :width_fullres // scale_ratio, :]
    cnn_out = tf.image.resize(cnn_out,
                              size=[height_fullres, width_fullres],
                              method='bilinear',
                              preserve_aspect_ratio=False,
                              antialias=True)
    return cnn_out[0]


class KerasAug:
    def __init__(self,
                 hue_delta: float,
                 saturation_range: List[float],
                 contrast_range: List[float],
                 brightness_delta: float,
                 zoom_range: List[float],
                 rotation_range: List[float],
                 ):
        self.hue_delta = hue_delta
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        with tf.device('/device:GPU:0'):
            self.spatial_transform_model = tf.keras.Sequential([tf.keras.layers.RandomZoom(height_factor=self.zoom_range,
                                                                                           width_factor=None,
                                                                                           fill_mode='reflect',
                                                                                           interpolation='bilinear'),
                                                                tf.keras.layers.RandomRotation(factor=self.rotation_range,
                                                                                               fill_mode='reflect',
                                                                                               interpolation='bilinear')
                                                                ])

    @tf.function
    def apply_color_transform(self, images):
        seed = tf.random.uniform(shape=[2], minval=-1000, maxval=1000, dtype=tf.int32)

        def color_transform(image):
            image = tf.image.stateless_random_hue(image, self.hue_delta, seed)
            image = tf.image.stateless_random_saturation(image, self.saturation_range[0], self.saturation_range[1],
                                                         seed)
            image = tf.image.stateless_random_contrast(image, self.contrast_range[0], self.contrast_range[1], seed)
            image = tf.image.stateless_random_brightness(image, self.brightness_delta, seed)
            return image

        images = tf.map_fn(color_transform, images)

        return images


    @tf.function
    def apply_spatial_transform(self, images):
        images_shape = tf.shape(images)
        images_num = images_shape[0]
        images = tf.concat(tf.unstack(images), axis=-1)
        with tf.device('/device:GPU:0'):
            images = self.spatial_transform_model(images)
        images = tf.stack(tf.map_fn(lambda x: images[:, :, x*3:x*3+3], tf.range(images_num), fn_output_signature=tf.float32))
        return images

    @tf.function
    def __call__(self, images):
        """
        images: 4D tensor [input, gt, ...]
        """
        images = self.apply_spatial_transform(images)
        images = self.apply_color_transform(images)
        return images
