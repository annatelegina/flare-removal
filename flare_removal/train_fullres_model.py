#!/usr/bin/env python
# coding: utf-8
import os
import time
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from train_utils.utils.wrappers import DataGenWrapper

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import albumentations as A
import cv2

from train_utils.train import losses, train_step
from train_utils.utils import dataset_tf, path_processors
from flare_removal.data_config import LOG_DIR, TRAIN_FOLDERS # revert after test
import flare_removal.models as models # revert after test
from train_utils.train.train_factory import AbstractTrainFactory
from train_utils.train.train_setup import check_repo, setup_gpu

MODEL_NAME = "test"
IMAGE_SHAPE = np.array([3648, 2736])

CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints", "fullres", MODEL_NAME)
if not os.path.isdir(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard", "fullres", MODEL_NAME)
if not os.path.isdir(TENSORBOARD_DIR):
    os.mkdir(TENSORBOARD_DIR)

WEIGHTS_DIR = os.path.join(LOG_DIR, "weights", "fullres", MODEL_NAME)
if not os.path.isdir(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)


class FullresTrainFactory(AbstractTrainFactory):

    @staticmethod
    def make_loss():
        VGG_loss = losses.make_VGG16_loss(blocks_dict={1: 1, 3: 1},
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
    
    def make_adversarial_loss():
        VGG_loss = losses.make_VGG16_loss(blocks_dict={1: 1, 3: 1},
                                            weights_path=os.path.abspath(
                                                r"train_utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                                            loss_type='MAE')
        
        LAM = 20
        adversarial_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

        def generator_loss(disc_logits, target, gen_output):
            l1_loss_val = losses.L1_loss(target, gen_output)
            vgg_loss_val = VGG_loss(target, gen_output)
            rec_loss = 0.5 * l1_loss_val + 4e-3 * vgg_loss_val

            adv_loss = adversarial_loss(tf.ones_like(disc_logits), disc_logits)

            total_loss = adv_loss + LAM * rec_loss

            return total_loss, {"l1_loss": l1_loss_val, "vgg_loss": vgg_loss_val, "adv_loss": adv_loss}

        def discriminator_loss(disc_real, disc_gen):
            real_loss = adversarial_loss(tf.ones_like(disc_real), disc_real)
            fake_loss = adversarial_loss(tf.zeros_like(disc_gen), disc_gen)
            total_loss = 0.5 * (real_loss + fake_loss)

            return total_loss
        
        return generator_loss, discriminator_loss

    @staticmethod
    def make_optimizer():
        LEARNING_RATE = 2e-4

        radam_g = tfa.optimizers.RectifiedAdam(learning_rate = LEARNING_RATE)
        optimizer_g = tfa.optimizers.Lookahead(radam_g, sync_period = 6, slow_step_size = 0.5)

        radam_d = tfa.optimizers.RectifiedAdam(learning_rate = LEARNING_RATE)
        optimizer_d = tfa.optimizers.Lookahead(radam_d, sync_period = 6, slow_step_size = 0.5)

        return optimizer_g, optimizer_d

    @staticmethod
    def make_model(data_format: models.DataFormat):
        generator = models.get_tiny_fullres_net(data_format = data_format)
        discriminator = models.get_patch_gan()
        return generator, discriminator

    @staticmethod
    def make_checkpoint(optimizer_g, generator, optimizer_d, discriminator):
        checkpoint = tf.train.Checkpoint(generator_optimizer = optimizer_g, generator = generator,\
                                         discriminator_optimizer = optimizer_d, discriminator = discriminator)
        return checkpoint
    
    @staticmethod
    def make_logger(sha: str) -> tf.summary.SummaryWriter:
        logger = tf.summary.create_file_writer(os.path.join(TENSORBOARD_DIR, sha))
        return logger

    @staticmethod
    def make_augmentations():
        transforms = A.Compose(
        [A.ShiftScaleRotate(scale_limit = [-0.25, 1.5], rotate_limit = 180, interpolation = cv2.INTER_CUBIC,
                            border_mode = cv2.BORDER_REFLECT_101, p = 0.8),
         A.HueSaturationValue(hue_shift_limit = (20 / 255), sat_shift_limit = (20 / 255), val_shift_limit = (30 / 255),
                              always_apply = False, p = 0.25),
         A.ChannelShuffle(p = 0.25)],
        additional_targets = {'image0': 'image', 'image1': 'image'})

        return transforms

    @staticmethod
    def make_dataset(transforms):
        PATCH_SIZE = 512
        BATCH_SIZE = 8

        path_processor = path_processors.FullresPathProcessor(transforms=transforms,
                                                        min_alpha=1,
                                                        r=100,)

        raw_generator = dataset_tf.get_dataset(data_paths=TRAIN_FOLDERS)
        dataset = DataGenWrapper.from_paths(raw_generator, path_processor, PATCH_SIZE, BATCH_SIZE)

        return dataset 

    @staticmethod
    def make_train_step():
        return train_step.make_train_step_PatchGAN()


def fit(epochs,
        G_loss,
        D_loss,
        optimizer_g,
        generator,
        optimizer_d,
        discriminator,
        train_step,
        dataset,
        checkpoint,
        logger,
        sha,
        weights_freq,
        ckpt_freq,
        images_freq,
        ckpt_path = None):

    if ckpt_path is not None:
        checkpoint.restore(ckpt_path)
        print("Training restored.")
    
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        gen_total_loss = 0
        disc_total_loss = 0

        data_generator = dataset.get_tf_dataset()
        for n, (input_image, target, alpha) in enumerate(data_generator):

            print('.', end='')
            sys.stdout.flush()

            if (n + 1) % 100 == 0:
                print('\n')

            losses_values = train_step(input_image = input_image,
                                       target = target,
                                       generator = generator,
                                       discriminator = discriminator,
                                       generator_loss = G_loss,
                                       discriminator_loss = D_loss,
                                       generator_optimizer = optimizer_g,
                                       discriminator_optimizer = optimizer_d,
                                       training = True,
                                       fullres = True)

            gen_total_loss += losses_values[0]
            disc_total_loss += losses_values[2]

            if n == 0:
                separate_losses = {key: 0 for key in losses_values[1].keys()}

            for key in losses_values[1].keys():
                separate_losses[key] += losses_values[1][key]

        with logger.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss / len(data_generator), step = epoch)
            tf.summary.scalar('disc_total_loss', disc_total_loss / len(data_generator), step = epoch)
            tf.summary.scalar('total_loss', (gen_total_loss + disc_total_loss) / len(data_generator), step = epoch)

            for key in separate_losses.keys():
                tf.summary.scalar(key, separate_losses[key] / len(data_generator), step = epoch)

        if (epoch + 1) % weights_freq == 0:
            generator.save(os.path.join(WEIGHTS_DIR, sha, "gen", "{}_epoch_{}_model.hdf5".format(sha, epoch)))
            discriminator.save(os.path.join(WEIGHTS_DIR, sha, "disc", "{}_epoch_{}_model.hdf5".format(sha, epoch)))

        if (epoch + 1) % images_freq == 0:
            preds = generator(input_image)
            with logger.as_default():
                tf.summary.image("Input_image", input_image[..., :3], step = epoch)
                tf.summary.image("Guidance", input_image[..., 3:], step = epoch)
                tf.summary.image("Ground truth", target, step = epoch)
                tf.summary.image("Model results", preds, step = epoch)

        if (epoch + 1) % ckpt_freq == 0:
            checkpoint.save(file_prefix = os.path.join(CHECKPOINT_DIR, sha, "ckpt"))

        print('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type = int, default = 0, help = 'Training GPU')
    parser.add_argument('--epochs', type = int, default = 10000, help = 'Number of epochs')
    parser.add_argument('--ckpt_freq', type = int, default = 100, help = 'Frequency of checkpoint saving')
    parser.add_argument('--weights_freq', type = int, default = 100, help = 'Frequency of weights saving')
    parser.add_argument('--images_freq', type = int, default = 50, help = 'Frequency of images saving')
    parser.add_argument('--ckpt', type = str, default = None, help = 'Path to checkpoint')

    args = parser.parse_args()

    sha = check_repo()

    setup_gpu(args.GPU)

    """Training initialization"""

    generator, discriminator = FullresTrainFactory.make_model(data_format = models.DataFormat.NHWC)
    generator_loss, discriminator_loss = FullresTrainFactory.make_adversarial_loss()
    optimizer_g, optimizer_d = FullresTrainFactory.make_optimizer()
    checkpoint = FullresTrainFactory.make_checkpoint(optimizer_g, generator, optimizer_d, discriminator)
    logger = FullresTrainFactory.make_logger(sha)

    train_step_function = FullresTrainFactory.make_train_step()

    print("Train is ready, building train data generator...")

    transforms = FullresTrainFactory.make_augmentations()
    dataset = FullresTrainFactory.make_dataset(transforms)

    if not os.path.isdir(os.path.join(WEIGHTS_DIR, sha)):
        os.mkdir(os.path.join(WEIGHTS_DIR, sha))

    print("Starting training process...")
    fit(epochs = args.epochs,
        G_loss = generator_loss,
        D_loss = discriminator_loss,
        optimizer_g = optimizer_g,
        generator = generator,
        optimizer_d = optimizer_d,
        discriminator = discriminator,
        train_step = train_step_function,
        dataset = dataset,
        checkpoint = checkpoint,
        logger = logger,
        sha = sha,
        weights_freq = args.weights_freq,
        ckpt_freq = args.ckpt_freq,
        images_freq = args.images_freq,
        ckpt_path = args.ckpt)