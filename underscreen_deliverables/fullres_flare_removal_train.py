#!/usr/bin/env python
# coding: utf-8

import sys
from enum import Enum

import tensorflow as tf
import tensorflow_addons as tfa

import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm
import cv2
from guided_filter_tf.guided_filter import guided_filter

from train_utils.utils import dataset, image_io
from train_utils.train import losses, train_step

CHECKPOINT_DIR = r"path_to_checkpoint_dir"
LOG_DIR = r"path_to_log_dir"
WEIGHTS_DIR = r"path_to_weights"
MODEL_SUFFIX = r"_model_suffix"
DOWNSCALE_GEN_WEIGHTS = r"path_to_weights"
DOWNSCALE_RATIO = 4
IMAGE_HEIGHT, IMAGE_WIDTH = 1080, 1920


class DatasetMode(Enum):
    IMAGES = 0
    PROCESSED = 1  # flare only datasets have horizontal orientation


# Create training dataset

DATASET_MODE = DatasetMode.IMAGES

if DATASET_MODE == DatasetMode.IMAGES:
    USC_FOLDERS = [r"usc_path_1",
                   r"usc_path_2..."]
    GT_FOLDERS = [r"gt_path_1",
                  r"gt_path_2..."]

    png_image_reader = image_io.ImageReaderPNG(transform_list=None)
    usc_arrays = dataset.get_images_array(folders_list=USC_FOLDERS, image_reader=png_image_reader,
                                          images_extension='.png')
    gt_arrays = dataset.get_images_array(folders_list=GT_FOLDERS, image_reader=png_image_reader,
                                         images_extension='.png')


    def img_resize(image):
        new_img = cv2.resize(image, (IMAGE_WIDTH // DOWNSCALE_RATIO, IMAGE_HEIGHT // DOWNSCALE_RATIO))
        return new_img


    usc_arrays_downscaled = [img_resize(image) for image in tqdm(usc_arrays)]
    gt_arrays_downscaled = [img_resize(image) for image in tqdm(gt_arrays)]

if DATASET_MODE == DatasetMode.PROCESSED:
    usc_arrays = np.load("usc_arrays_flare_only_full.npz")['arr_0']
    gt_arrays = np.load("gt_arrays_flare_only_full.npz")['arr_0']

    usc_arrays_downscaled = np.load("usc_arrays_flare_only.npz")['arr_0']
    gt_arrays_downscaled = np.load("gt_arrays_flare_only.npz")['arr_0']

# ## GPU selection

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

for num, (usc_image, gt_image) in tqdm(list(enumerate(zip(usc_arrays, gt_arrays)))):
    gt_image = guided_filter(x=tf.cast(gt_image[np.newaxis, :, :, :], tf.float32),
                             y=tf.cast(usc_image[np.newaxis, :, :, :], tf.float32), r=100, nhwc=True)
    gt_image = gt_image.numpy()[0]
    gt_arrays[num] = gt_image


# Build models

# Generator


def conv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', strides=strides,
                               data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def dilated_conv_block(growth_rate, filters, kernel_size, dilation_rate, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                               data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def conv_skip_block(growth_rate, filters, kernel_size, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same',
                                        data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def deconv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', strides=strides,
                                        data_format='channels_first')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 1, padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def Generator(height=None, width=None, input_channels=3, filters=16):
    inputs = tf.keras.Input(shape=[input_channels, height, width])

    x = dilated_conv_block(growth_rate=2, filters=filters, kernel_size=(5, 5), dilation_rate=8, x=inputs)
    res1 = x

    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format='channels_first')(x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res2 = x

    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format='channels_first')(x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)

    x = deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res2])
    x = conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)

    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)

    x = deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res1])
    x = conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)

    x = dilated_conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), dilation_rate=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.Subtract()([inputs[:, :3, :, :], x])

    _model = tf.keras.Model(inputs=inputs, outputs=x, name='derain_net')
    return _model


def Generator_downscale(height=None, width=None, input_channels=3, filters=16):
    inputs = tf.keras.Input(shape=[input_channels, height, width])

    x = dilated_conv_block(growth_rate=2, filters=filters, kernel_size=(9, 9), dilation_rate=16, x=inputs)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(5, 5), dilation_rate=8, x=x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res1 = x

    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format='channels_first')(x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res2 = x

    x = dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format='channels_first')(x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)

    x = deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res2])
    x = conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)

    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)

    x = deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res1])
    x = conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)

    x = dilated_conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), dilation_rate=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.Subtract()([inputs, x])
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', data_format='channels_first')(x)

    _model = tf.keras.Model(inputs=inputs, outputs=x, name='derain_net')
    return _model


generator_full = Generator(height=None, width=None, input_channels=6, filters=16)

# ## Guidance data preparation


guidance_cnn = Generator_downscale(input_channels=3, filters=32)
guidance_cnn.load_weights(DOWNSCALE_GEN_WEIGHTS)


def prepare_full_res_guidance(usc_downscaled_image, cnn):
    height, width = usc_downscaled_image.shape[:2]
    pad_0 = int((4 - height % 4) % 4)
    pad_1 = int((4 - width % 4) % 4)
    paddings = tf.constant([[0, 0], [0, pad_0], [0, pad_1], [0, 0]])
    padded_image = tf.cast(usc_downscaled_image[np.newaxis, :, :, :], tf.float32)
    padded_image = tf.pad(padded_image, paddings, "SYMMETRIC")
    cnn_out = cnn(tf.transpose(padded_image, [0, 3, 1, 2]))
    np_out = tf.transpose(cnn_out, [0, 2, 3, 1]).numpy()[0]
    np_out = np_out[(np.array(list(range(np_out.shape[0]-pad_0)))[:, np.newaxis],
                     np.array(list(range(np_out.shape[1]-pad_1))),
                     ...)]
    np_out = np.clip(np_out, 0, 1)
    np_out = cv2.resize(np_out, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np_out


usc_fullres_guidnance_arrays = [prepare_full_res_guidance(image, guidance_cnn) for image in tqdm(usc_arrays_downscaled)]
usc_concatenated_fullres_arrays = [np.dstack([usc_image, usc_guidance]) for
                                   (usc_image, usc_guidance) in tqdm(zip(usc_arrays, usc_fullres_guidnance_arrays))]

training_dataset = (usc_concatenated_fullres_arrays, gt_arrays)
train_data_generator = dataset.DataGenerator(training_dataset, patch_size=896, batch_size=1, shuffle=True, pad=11)

# ## Define losses

# ### Generator loss

vgg16_loss_1_3 = losses.make_VGG16_loss(blocks_dict={1: 1, 3: 1},
                                        weights_path=r"train_utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                        loss_type='MAE')


def generator_loss(gen_output, target):
    gen_output = tf.transpose(gen_output, [0, 2, 3, 1])
    target = tf.transpose(target, [0, 2, 3, 1])
    l1_loss_val = losses.L1_loss(target, gen_output)
    ms_ssim_loss_val = (1 - tf.math.reduce_mean(
        tf.image.ssim_multiscale(target, gen_output, 1, power_factors=(0.4, 0.25, 0.25, 0.2363, 0.1333))))
    vgg_loss_val = vgg16_loss_1_3(tf.image.resize(target, [224, 224]), tf.image.resize(gen_output, [224, 224]))
    total_gen_loss_val = 0.5 * l1_loss_val + 1 * ms_ssim_loss_val + 4e-3 * vgg_loss_val

    return total_gen_loss_val, [l1_loss_val, ms_ssim_loss_val, vgg_loss_val]


# Define the Optimizers and Checkpoint-saver

radam = tfa.optimizers.RectifiedAdam(lr=2e-4)
generator_optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

checkpoint_dir = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator_full)

# ## Training


summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR,
                                                            "fit",
                                                            datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX))

file_writer_img = tf.summary.create_file_writer(os.path.join(LOG_DIR,
                                                             "fit",
                                                             datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX,
                                                             "img"))

import albumentations as A

transform = A.Compose(
    [A.ShiftScaleRotate(scale_limit=[-0.01, 5], rotate_limit=180, interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101, p=0.8),
     A.HueSaturationValue(hue_shift_limit=20 / 255, sat_shift_limit=20 / 255, val_shift_limit=30 / 255,
                          always_apply=False, p=0.25),
     A.ChannelShuffle(p=0.25)],
    additional_targets={'image0': 'image', 'image1': 'image'}
)

train_step = train_step.make_train_step()


def fit(epochs):
    for epoch in range(epochs):
        start = time.time()

        if (epoch + 1) % 2 == 0:
            generator_full.save_weights(os.path.join(WEIGHTS_DIR, "fullres_weights_epoch_{}.hdf5".format(epoch)))
            with tf.device('/GPU:0'):
                tf.keras.backend.clear_session()
                test_generator = Generator(height=None, width=None, input_channels=6, filters=16)
                test_generator.load_weights(os.path.join(WEIGHTS_DIR, "fullres_weights_epoch_{}.hdf5".format(epoch)))
                img_index = np.random.randint(len(usc_concatenated_fullres_arrays))
                test_image = usc_concatenated_fullres_arrays[img_index][np.newaxis, :, :, :]
                test_image = tf.cast(test_image, tf.float32)
                preds = test_generator(tf.transpose(test_image, [0, 3, 1, 2]), training=True)
                preds = tf.transpose(preds, [0, 2, 3, 1])

            with file_writer_img.as_default():
                tf.summary.image("Input_image", usc_arrays[img_index][np.newaxis, :, :, :], step=epoch)
                tf.summary.image("Ground truth", gt_arrays[img_index][np.newaxis, :, :, :], step=epoch)
                tf.summary.image("Guidance", usc_fullres_guidnance_arrays[img_index][np.newaxis, :, :, :], step=epoch)
                tf.summary.image("Model results", preds.numpy()[:, :, :, :3], step=epoch)

        print("Epoch: ", epoch)
        train_data_generator.on_epoch_end()
        # Train
        gen_total_loss, l1_loss_val, ms_ssim_loss_val, vgg_loss_val = 0, 0, 0, 0

        for n, (input_image, target) in enumerate(train_data_generator):
            print('.', end='')
            sys.stdout.flush()
            if (n + 1) % 100 == 0:
                print('\n')
            input_image = np.array(input_image)
            target = np.array(target)

            for pair_id in range(len(input_image)):
                transformed = transform(image=input_image[pair_id][:, :, :3], image0=input_image[pair_id][:, :, 3:6],
                                        image1=target[pair_id])
                input_image[pair_id] = np.dstack([transformed['image'], transformed['image0']])
                target[pair_id] = transformed['image1']

            input_image = tf.cast(input_image, tf.float32)
            target = tf.cast(target, tf.float32)
            losses = train_step(input_image=tf.transpose(tf.cast(input_image, tf.float32), [0, 3, 1, 2]),
                                target=tf.transpose(target, [0, 3, 1, 2]),
                                generator=generator_full,
                                generator_loss=generator_loss,
                                generator_optimizer=generator_optimizer,
                                training=True)

            gen_total_loss += losses[0]

            l1_loss_val += losses[1][0]
            ms_ssim_loss_val += losses[1][1]
            vgg_loss_val += losses[1][2]

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss / len(train_data_generator), step=epoch)
            tf.summary.scalar('l1_loss_val', l1_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('ms_ssim_loss_val', ms_ssim_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('vgg_loss_val', vgg_loss_val / len(train_data_generator), step=epoch)

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('\n')
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))


EPOCHS = 10000
fit(EPOCHS)
