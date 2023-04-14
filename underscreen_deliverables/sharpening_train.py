#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
from datetime import datetime
from enum import Enum

import albumentations as A
import cv2
import git
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from IPython import display

from train_utils.utils import dataset, image_io

CHECKPOINT_DIR = r"path_to_checkpoint_dir"
LOG_DIR = r"path_to_log_dir"
WEIGHTS_DIR = r"path_to_weights"
MODEL_SUFFIX = r"_model_suffix"


class DatasetMode(Enum):
    IMAGES = 0
    PROCESSED = 1


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

if DATASET_MODE == DatasetMode.PROCESSED:
    usc_arrays = np.load("../20210817_lr_arrays_orig_faces.npz")['arr_0']
    gt_arrays = np.load("../20210817_gt_arrays_downtolr_faces.npz")['arr_0']

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

# In[39]:


tf.keras.backend.clear_session()

# In[7]:


training_dataset = (usc_arrays, gt_arrays)

train_data_generator = dataset.DataGenerator(training_dataset, patch_size=224, batch_size=8, shuffle=True,
                                             pad=2)

# ## Build the generator (custom resize changed to standart layers)


from train_utils.train import losses



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


# In[10]:


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
    x = tf.keras.layers.Subtract()([inputs, x])
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='derain_net')
    return _model


# In[11]:


OUTPUT_CHANNELS = 3
generator = Generator()
# generator.summary()


# ### Generator loss

# In[12]:


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# In[13]:


vgg16_loss_1_3 = losses.make_VGG16_loss(blocks_dict={1: 1, 2: 1, 3: 1},
                                        weights_path=r"train_utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                        loss_type='MAE')


# In[14]:


def generator_loss(gen_output, target):
    l1_loss_val = losses.L1_loss(target, gen_output)

    ms_ssim_loss_val = (1 - tf.math.reduce_mean(tf.image.ssim_multiscale(target, gen_output, 1, power_factors=(
        0.4, 0.25, 0.25, 0.5, 0.5))))

    vgg_loss_val = vgg16_loss_1_3(target, gen_output)

    total_gen_loss_val = 0.5 * l1_loss_val + 5 * ms_ssim_loss_val + 4e-3 * vgg_loss_val

    return total_gen_loss_val, [l1_loss_val, ms_ssim_loss_val, vgg_loss_val]


# In[15]:


def generator_loss_full(disc_generated_output, gen_output, target):
    gan_loss_val = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    total_gen_loss_val, [l1_loss_val, ms_ssim_loss_val, vgg_loss_val] = generator_loss(gen_output, target)

    total_gen_loss = 0.1 * gan_loss_val + total_gen_loss_val

    return total_gen_loss, [l1_loss_val, ms_ssim_loss_val, vgg_loss_val, gan_loss_val]


# ## Build the Discriminator

# In[16]:


def downsample(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.LeakyReLU())

    return result


# In[17]:


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    # batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(conv)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# In[18]:


discriminator = Discriminator()


# ### Discriminator loss

# In[19]:


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# ## Define the Optimizers and Checkpoint-saver

# In[61]:

initial_learning_rate_generator = 2e-4
initial_learning_rate_discriminator = 1e-4

# In[62]:


radam_gen = tfa.optimizers.RectifiedAdam(lr=initial_learning_rate_generator)
generator_optimizer = tfa.optimizers.Lookahead(radam_gen, sync_period=6, slow_step_size=0.5)

radam_disc = tfa.optimizers.RectifiedAdam(lr=initial_learning_rate_discriminator)
discriminator_optimizer = tfa.optimizers.Lookahead(radam_disc, sync_period=6, slow_step_size=0.5)

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# In[24]:

checkpoint_dir = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# ## Training

summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR,
                                                            "fit",
                                                            datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX))

file_writer_img = tf.summary.create_file_writer(os.path.join(LOG_DIR,
                                                             "fit",
                                                             datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_SUFFIX,
                                                             "img"))


@tf.function
def train_step_PatchGAN(y_lr_input,
                        y_hr_input,
                        uv_lr_input,
                        generator,
                        discriminator,
                        generator_loss,
                        discriminator_loss,
                        generator_optimizer,
                        discriminator_optimizer,
                        training=True):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(y_lr_input, training=training)

        gen_output = tf.image.yuv_to_rgb(tf.concat([gen_output, uv_lr_input], axis=-1))
        target = tf.image.yuv_to_rgb(tf.concat([y_hr_input, uv_lr_input], axis=-1))
        input_image = tf.image.yuv_to_rgb(tf.concat([y_lr_input, uv_lr_input], axis=-1))

        disc_real_output = discriminator([input_image, target], training=training)
        disc_generated_output = discriminator([input_image, gen_output], training=training)

        gen_total_loss, gen_losses = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    return gen_total_loss, gen_losses, disc_loss


# In[64]:


transform = A.Compose(
    [A.ShiftScaleRotate(scale_limit=[-0.9, 1.1], rotate_limit=180, interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101, p=0.7)],
    additional_targets={'image0': 'image'}
)


def get_train_data(lr_image, hr_image):
    y_lr_channel = tf.image.rgb_to_yuv(lr_image)[..., 0:1]
    y_hr_channel = tf.image.rgb_to_yuv(hr_image)[..., 0:1]
    uv_lr_channels = tf.image.rgb_to_yuv(lr_image)[..., 1:3]
    return y_lr_channel, y_hr_channel, uv_lr_channels


# In[65]:


def fit(epochs, run):
    for epoch in range(epochs):
        epoch += int(epochs * run)
        start = time.time()
        display.clear_output(wait=True)

        if (epoch + 1) % 5 == 0:
            generator.save_weights(os.path.join(WEIGHTS_DIR, "sharpening_weights_epoch_{}.hdf5".format(epoch)))
            with tf.device('/CPU:0'):
                tf.keras.backend.clear_session()
                test_generator = Generator()
                test_generator.load_weights(os.path.join(WEIGHTS_DIR, "sharpening_weights_epoch_{}.hdf5".format(epoch)))
                img_index = np.random.randint(len(usc_arrays))
                coords = dataset.get_random_corner(usc_arrays[img_index].shape[:2], 512, 25)
                test_image = dataset.get_patch(usc_arrays[img_index],
                                               coords,
                                               512)[np.newaxis, ...]
                y_test_image = tf.image.rgb_to_yuv(test_image)[..., 0:1]
                uv_test_image = tf.image.rgb_to_yuv(test_image)[..., 1:3]
                preds = test_generator(y_test_image, training=True)
                preds = tf.concat([preds, uv_test_image], axis=-1)
                preds = tf.image.yuv_to_rgb(preds)

            with file_writer_img.as_default():
                tf.summary.image("Input_image", dataset.get_patch(usc_arrays[img_index],
                                                                  coords,
                                                                  512)[np.newaxis, ...],
                                 step=epoch)
                tf.summary.image("Ground truth", dataset.get_patch(gt_arrays[img_index],
                                                                   coords,
                                                                   512)[np.newaxis, ...],
                                 step=epoch)
                tf.summary.image("Model results", preds.numpy()[:, :, :, :3], step=epoch)

        print("Epoch: ", epoch)

        train_data_generator.on_epoch_end()
        # Train
        gen_total_loss, l1_loss_val, ms_ssim_loss_val, vgg_loss_val, gan_loss_val, disc_loss_val = 0, 0, 0, 0, 0, 0

        for n, (input_image, target) in enumerate(train_data_generator):
            print('.', end='')
            sys.stdout.flush()
            if (n + 1) % 100 == 0:
                print('\n')
            input_image = np.array(input_image)
            target = np.array(target)

            for pair_id in range(len(input_image)):
                transformed = transform(image=input_image[pair_id], image0=target[pair_id])

                input_image[pair_id] = transformed['image']
                target[pair_id] = transformed['image0']

            input_image = tf.cast(input_image, tf.float32)
            target = tf.cast(target, tf.float32)

            y_lr, y_hr, uv_lr = get_train_data(input_image, target)

            train_losses = train_step_PatchGAN(y_lr_input=tf.cast(y_lr, tf.float32),
                                               y_hr_input=tf.cast(y_hr, tf.float32),
                                               uv_lr_input=tf.cast(uv_lr, tf.float32),
                                               generator=generator,
                                               discriminator=discriminator,
                                               generator_loss=generator_loss_full,
                                               discriminator_loss=discriminator_loss,
                                               generator_optimizer=generator_optimizer,
                                               discriminator_optimizer=discriminator_optimizer,
                                               training=True)

            gen_total_loss += train_losses[0]

            l1_loss_val += train_losses[1][0]
            ms_ssim_loss_val += train_losses[1][1]
            vgg_loss_val += train_losses[1][2]
            gan_loss_val += train_losses[1][3]

            disc_loss_val += train_losses[2]

        print()
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss / len(train_data_generator), step=epoch)
            tf.summary.scalar('l1_loss_val', l1_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('ms_ssim_loss_val', ms_ssim_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('vgg_loss_val', vgg_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('gan_loss_val', gan_loss_val / len(train_data_generator), step=epoch)
            tf.summary.scalar('disc_loss_val', disc_loss_val / len(train_data_generator), step=epoch)

        # saving (checkpoint) the model every 250 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))


# In[66]:


EPOCHS = 1000

# In[67]:


for i in range(20):
    fit(EPOCHS, i)
