#!/usr/bin/env python
# coding: utf-8
import os
import time
import sys
import argparse
from pathlib import Path

from train_utils.utils.wrappers import DataGenWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import albumentations as A
import cv2

sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_utils.utils.augmentations import FlareAdder, FlareAug, ZeroChannelAug
from train_utils.train import losses, train_step
from train_utils.utils import dataset_tf, path_processors
from data_config import LOG_DIR, TRAIN_FOLDERS
import models
from train_utils.train.train_factory import AbstractTrainFactory
from train_utils.train.train_setup import check_repo, setup_gpu

CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
if not os.path.isdir(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
if not os.path.isdir(TENSORBOARD_DIR):
    os.mkdir(TENSORBOARD_DIR)

WEIGHTS_DIR = os.path.join(LOG_DIR, "weights")
if not os.path.isdir(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

IMAGE_SHAPE = np.array([3648, 2736])


class DownscaleTrainFactory(AbstractTrainFactory):

    @staticmethod
    def make_loss():
        VGG_loss = losses.make_VGG16_loss(blocks_dict={1: 1, 2:1, 3: 1},
                                            weights_path=os.path.abspath(
                                                r"train_utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                                            loss_type='MAE')
        def generator_loss(gen_output, target):
            l1_loss_val = losses.L1_loss(target, gen_output)
            ms_ssim_loss_val = (1 - (1 + tf.math.reduce_mean(tf.image.ssim_multiscale(target,
                                                                                      gen_output,
                                                                                      1,
                                                                                      power_factors=(
                                                                                          0.4, 0.25, 0.25,
                                                                                          0.2363)))) / 2)  # , 0.1333))))
            vgg_loss_val = VGG_loss(target, gen_output)
            total_gen_loss_val = 0.5 * l1_loss_val + 0 * 1 * ms_ssim_loss_val + 4e-3 * vgg_loss_val

            return total_gen_loss_val, {"l1_loss": l1_loss_val, "ms_ssim_loss": ms_ssim_loss_val,
                                        "vgg_loss": vgg_loss_val}

        return generator_loss

    @staticmethod
    def make_optimizer():
        LEARNING_RATE=2e-4

        radam = tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE)
        generator_optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        return generator_optimizer

    @staticmethod
    def make_model(data_format: models.DataFormat):
        generator = models.get_vanilla_downscale_net(data_format=data_format)
        return generator

    @staticmethod
    def make_checkpoint(optimizer, model):
        checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer, generator=model)
        return checkpoint

    @staticmethod
    def make_metrics_writer(sha):
        metrics_writer = tf.summary.create_file_writer(os.path.join(TENSORBOARD_DIR, "fit", sha))
        return metrics_writer

    @staticmethod
    def make_files_writer(sha):
        files_writer = tf.summary.create_file_writer(os.path.join(TENSORBOARD_DIR, "fit", sha, "img"))
        return files_writer

    @staticmethod
    def make_augmentations():
        flare_transforms = A.Compose(
            [A.ShiftScaleRotate(rotate_limit=180, interpolation=cv2.INTER_CUBIC,
                                border_mode=cv2.BORDER_REFLECT_101, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20 / 255, sat_shift_limit=20 / 255, val_shift_limit=30 / 255,
                                always_apply=False, p=0.25),
            A.ChannelShuffle(p=0.25),
            ZeroChannelAug()
            ],
            additional_targets={'image0': 'image'})

        common_transforms = A.Compose(
            [A.ShiftScaleRotate(scale_limit=[-0.65, 0.25], rotate_limit=90, interpolation=cv2.INTER_CUBIC,
                                border_mode=cv2.BORDER_REFLECT_101, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20 / 255, sat_shift_limit=20 / 255, val_shift_limit=30 / 255,
                                always_apply=False, p=0.25),
            A.ChannelShuffle(p=0.25),
            ZeroChannelAug(p=0.25)
            ],
            additional_targets={'image0': 'image'})

        transforms = A.Compose(
            [
                # FlareAug(FlareAdder('data/20220603_dark_background/', flare_transforms), min_size=32, p=.1),
                common_transforms
            ])

        return transforms

    @staticmethod
    def make_dataset(transforms):
        PATCH_SIZE = 324
        BATCH_SIZE = 4

        path_processor = path_processors.BasePathProcessor(resize_shape=IMAGE_SHAPE // 8,
                                                       transforms=transforms,
                                                       min_alpha=1,
                                                       r=100)

        raw_generator = dataset_tf.get_dataset(data_paths=TRAIN_FOLDERS)
        dataset = DataGenWrapper.from_paths(raw_generator, path_processor, PATCH_SIZE, BATCH_SIZE)

        return dataset

    @staticmethod
    def make_train_step():
        return train_step.make_train_step()

def fit(epochs,
        loss,
        optimizer,
        generator,
        train_step,
        dataset,
        checkpoint,
        metrics_writer,
        images_writer,
        sha,
        weights_freq,
        ckpt_freq,
        images_freq):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        gen_total_loss = 0
        data_generator = dataset.get_tf_dataset()
        for n, (input_image, target, alpha) in enumerate(data_generator):

            print('.', end='')
            sys.stdout.flush()

            if (n + 1) % 100 == 0:
                print('\n')

            losses_values = train_step(input_image=input_image,
                                       target=target,
                                       generator=generator,
                                       generator_loss=loss,
                                       generator_optimizer=optimizer,
                                       training=True)

            gen_total_loss += losses_values[0]

            if n == 0:
                separate_losses = {key: 0 for key in losses_values[1].keys()}

            for key in losses_values[1].keys():
                separate_losses[key] += losses_values[1][key]

        with metrics_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss / len(data_generator), step=epoch)

            for key in separate_losses.keys():
                tf.summary.scalar(key, separate_losses[key] / len(data_generator), step=epoch)

        if (epoch + 1) % weights_freq == 0:
            generator.save(os.path.join(WEIGHTS_DIR, sha, "{}_epoch_{}_model.hdf5".format(sha, epoch)))

        if (epoch + 1) % images_freq == 0 or epoch == 0: # We want to validate input from first batches
            preds = generator(input_image)
            with images_writer.as_default():
                tf.summary.image("Input_image", input_image, step=epoch)
                tf.summary.image("Ground truth", target, step=epoch)
                tf.summary.image("Model results", preds, step=epoch)

        if (epoch + 1) % ckpt_freq == 0:
            checkpoint.save(file_prefix=os.path.join(CHECKPOINT_DIR, sha, "ckpt"))

        dataset.on_epoch_end()

        print('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                             time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=int, default=0, help='Training GPU')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--ckpt_freq', type=int, default=50, help='Frequency of checkpoint saving')
    parser.add_argument('--weights_freq', type=int, default=50, help='Frequency of weights saving')
    parser.add_argument('--images_freq', type=int, default=50, help='Frequency of images saving')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')

    args = parser.parse_args()

    if not args.debug:
        sha = check_repo()
    else:
        print('\033[93m'+'WARNING: debug mode'+'\033[0m', file=sys.stderr) # Printing in yellow
        sha = 'debug'

    setup_gpu(args.GPU)

    if not os.path.isdir(os.path.join(WEIGHTS_DIR, sha)):
        os.mkdir(os.path.join(WEIGHTS_DIR, sha))

    """Training initialization"""

    

    generator = DownscaleTrainFactory.make_model(data_format=models.DataFormat.NHWC)
    generator_loss = DownscaleTrainFactory.make_loss()
    optimizer = DownscaleTrainFactory.make_optimizer()
    checkpoint = DownscaleTrainFactory.make_checkpoint(optimizer, generator)
    metrics_writer = DownscaleTrainFactory.make_metrics_writer(sha)
    images_writer = DownscaleTrainFactory.make_files_writer(sha)

    #checkpoint.restore(os.path.join(CHECKPOINT_DIR,r"6f717c90f32d6903234c27a78a00a191ff4449f4", r"ckpt-36"))

    train_step_function = DownscaleTrainFactory.make_train_step()

    print("Train is ready, building train data generator...")
    transforms = DownscaleTrainFactory.make_augmentations()
    dataset = DownscaleTrainFactory.make_dataset(transforms)
    
    print("Starting training process...")
    fit(epochs=args.epochs,
        loss=generator_loss,
        optimizer=optimizer,
        generator=generator,
        train_step=train_step_function,
        dataset=dataset,
        checkpoint=checkpoint,
        metrics_writer=metrics_writer,
        images_writer=images_writer,
        sha=sha,
        weights_freq=args.weights_freq,
        ckpt_freq=args.ckpt_freq,
        images_freq=args.images_freq)
