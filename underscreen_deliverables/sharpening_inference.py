#!/usr/bin/env python
# coding: utf-8


import os

import numpy as np
import skimage.filters
import skimage.io
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


# ## GPU selection



# ## Initialize models

# In[10]:


def conv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', strides=strides, )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def dilated_conv_block(growth_rate, filters, kernel_size, dilation_rate, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', dilation_rate=dilation_rate, )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def conv_skip_block(growth_rate, filters, kernel_size, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def deconv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', strides=strides, )(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 1, padding='same', )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def BAM_block(x, channels):
    ch_a = tf.math.reduce_mean(x, axis=(1, 2), keepdims=True)
    ch_a = tf.keras.layers.Conv2D(filters=channels // 2, kernel_size=1, use_bias=False, activation='relu')(ch_a)
    ch_a = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=False, activation='sigmoid')(ch_a)

    sp_a = tf.math.reduce_max(x, axis=-1, keepdims=True)
    sp_a = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='same', use_bias=False, activation='sigmoid')(sp_a)

    b_a = ch_a * sp_a
    return b_a


# In[11]:


def Generator(height=None, width=None, input_channels=1, filters=32):
    inputs = tf.keras.Input(shape=[height, width, input_channels])

    x = conv_block(growth_rate=2, filters=filters, kernel_size=(5, 5), strides=1, x=inputs)
    x = x * BAM_block(x, 2 * filters)
    res1 = x

    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=2, x=x)
    x = x * BAM_block(x, 4 * filters)
    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = x * BAM_block(x, 4 * filters)
    res2 = x

    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=2, x=x)
    x = x * BAM_block(x, 4 * filters)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = x * BAM_block(x, 8 * filters)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = x * BAM_block(x, 8 * filters)

    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=8, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=16, x=x)
    x = x * BAM_block(x, 8 * filters)

    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = x * BAM_block(x, 8 * filters)

    x = deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)
    x = x * BAM_block(x, 4 * filters)

    x = tf.keras.layers.Concatenate(axis=-1)([x, res2])
    x = conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)

    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = x * BAM_block(x, 8 * filters)

    x = deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)
    x = x * BAM_block(x, 2 * filters)

    x = tf.keras.layers.Concatenate(axis=-1)([x, res1])
    x = conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)

    x = conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = tf.keras.layers.Conv2D(1, (3, 3), padding='same', )(x)
    x = tf.keras.layers.Subtract()([inputs[..., 0:1], x])
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='derain_net')
    return _model


def prepare_full_res_guidance(usc_downscaled_image, cnn):
    height, width = usc_downscaled_image.shape[:2]
    pad_0 = int((4 - height % 4) % 4)
    pad_1 = int((4 - width % 4) % 4)
    paddings = tf.constant([[0, 0], [0, pad_0], [0, pad_1], [0, 0]])
    padded_image = tf.cast(usc_downscaled_image[np.newaxis, :, :, :], tf.float32)
    padded_image = tf.pad(padded_image, paddings, "SYMMETRIC")
    padded_image_y = tf.image.rgb_to_yuv(padded_image)[..., 0:1]
    padded_image_uv = tf.image.rgb_to_yuv(padded_image)[..., 1:3]
    cnn_out = cnn(padded_image_y)
    result_rgb = tf.image.yuv_to_rgb(tf.concat([cnn_out, padded_image_uv], axis=-1)).numpy()[0]
    np_out = result_rgb[(np.array(list(range(result_rgb.shape[0] - pad_0)))[:, np.newaxis],
                         np.array(list(range(result_rgb.shape[1] - pad_1))),
                         ...)]
    np_out = np.clip(np_out, 0, 1)
    return np_out


def get_rgb_image(png_image_path):
    rgb_image_array = skimage.io.imread(png_image_path)
    if rgb_image_array.max() > 1:
        rgb_image_array = rgb_image_array / 255
    rgb_image_array = np.clip(rgb_image_array, 0, 1)
    return rgb_image_array[:, :, :3]


def get_output(input_folder, output_folder, model_weights_path, gpu_index, use_cpu):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_index], True)
    tf.config.experimental.set_visible_devices(physical_devices[gpu_index], 'GPU')

    sharpening_model = Generator(height=None, width=None, input_channels=1, filters=32)
    sharpening_model.load_weights(model_weights_path)

    usc_filenames = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    usc_filenames.sort()
    for filename in tqdm(usc_filenames):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            usc_image = get_rgb_image(filename)
            if use_cpu:
                with tf.device("/CPU:0"):
                    processed_image = prepare_full_res_guidance(usc_image, sharpening_model)
            else:
                processed_image = prepare_full_res_guidance(usc_image, sharpening_model)
            plt.imsave(os.path.join(output_folder, os.path.splitext(os.path.basename(filename))[0]+r"_SR.png"),
                       np.clip(processed_image, 0, 1))
            plt.imsave(os.path.join(output_folder, os.path.splitext(os.path.basename(filename))[0] + r"_LR.png"),
                       np.clip(usc_image, 0, 1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=r"input_images")
    parser.add_argument("--output_folder", default=r"output_images")
    parser.add_argument("--model_weights", default=r"197cf30558a5b15cb3c7b76eb3918a3ffcca480e_2849.hdf5")
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--use_cpu", type=int, default=0)

    parser_args = parser.parse_args()

    get_output(parser_args.input_folder,
               parser_args.output_folder,
               parser_args.model_weights,
               parser_args.GPU, parser_args.use_cpu)
