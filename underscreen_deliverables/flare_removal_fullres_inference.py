import os
import time
import glob
import sys

import numpy as np
from tqdm import tqdm
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

DOWNSCALE_RATIO = 4
IMAGE_HEIGHT, IMAGE_WIDTH = 1920, 1080

downscale_model_weights = r"20210303_dilation_stack_activations_shift.hdf5"
full_res_model_weights = r"20210317_full_res_shift_optimized.hdf5"

CHANGE_VIDEO_ORIENTATION = True  # required if train was in horizontal orientation
STACK = True  # stack input and output videos


def conv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', strides=strides, data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def dilated_conv_block(growth_rate, filters, kernel_size, dilation_rate, x):
    x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', dilation_rate=dilation_rate, data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def conv_skip_block(growth_rate, filters, kernel_size, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def deconv_block(growth_rate, filters, kernel_size, strides, x):
    x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', strides=strides, data_format='channels_first')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 1, padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def build_derain_generator(height=None, width=None, input_channels=3, filters=32):
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


def build_derain_generator_2(height=None, width=None, input_channels=6, filters=16):
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


def prepare_full_res_guidance(usc_downscaled_image, cnn):
    if CHANGE_VIDEO_ORIENTATION:
        usc_downscaled_image = cv2.rotate((usc_downscaled_image*255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)/255
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
    if CHANGE_VIDEO_ORIENTATION:
        np_out = cv2.rotate((np_out*255).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)/255
    np_out = cv2.resize(np_out, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np_out


def get_full_res_output(image, cnn):
    if CHANGE_VIDEO_ORIENTATION:
        image = cv2.rotate((image*255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)/255
    test_image = image[np.newaxis, :, :, :]
    test_image = tf.cast(test_image, tf.float32)
    preds = cnn(tf.transpose(test_image, [0, 3, 1, 2]), training=True)
    preds = tf.transpose(preds, [0, 2, 3, 1])
    preds = preds.numpy()[0]
    preds = np.clip(preds, 0, 1)
    if CHANGE_VIDEO_ORIENTATION:
        preds = cv2.rotate((preds*255).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)/255
    return preds


def img_resize(image):
    new_img = cv2.resize(image, (IMAGE_WIDTH//DOWNSCALE_RATIO, IMAGE_HEIGHT//DOWNSCALE_RATIO))
    return new_img


def process_video(video_input_folder, video_out_folder):
    downscale_model = build_derain_generator(height=None, width=None, input_channels=3, filters=32)
    full_res_model = build_derain_generator_2(height=None, width=None, input_channels=6, filters=16)

    downscale_model.load_weights(downscale_model_weights)
    full_res_model.load_weights(full_res_model_weights)

    for usc_video_path in sorted(glob.glob(video_input_folder + r"/*.mp4")):
        video_name = os.path.split(usc_video_path)[-1][:-4] + '.avi'
        print("Processing video {}".format(video_name))
        video_cap_usc = cv2.VideoCapture(usc_video_path)
        success = True
        i = 0
        while success:
            success, frame = video_cap_usc.read()
            if success:
                frame = np.array(frame) / 255
                frame = frame[:, :, ::-1].astype(np.float32)
                if i == 0:
                    frame_shape = frame.shape[0:2]
                    out = cv2.VideoWriter(os.path.join(video_out_folder, video_name),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                          (frame_shape[1] * 2 if STACK else frame_shape[1], frame_shape[0]))
                print('.', end='')
                sys.stdout.flush()
                if (i + 1) % 100 == 0:
                    print('\n')
                downscaled_frame = img_resize(frame)
                fullres_guidance_frame = prepare_full_res_guidance(downscaled_frame, downscale_model)
                concatenated_fullres_frame = np.dstack([frame, fullres_guidance_frame])
                full_res_output_frame = get_full_res_output(concatenated_fullres_frame, full_res_model)
                out_image = full_res_output_frame

                out_image = np.clip(out_image, 0, 1)
                out_image = out_image * 255
                out_image = out_image.astype(np.uint8)

                usc_image = np.clip(frame, 0, 1)
                usc_image = usc_image * 255
                usc_image = usc_image.astype(np.uint8)

                triple_img = np.hstack([usc_image, out_image])
                if STACK:
                    out.write(triple_img[:, :, ::-1])
                else:
                    out.write(out_image[:, :, ::-1])
                i += 1
        out.release()
        video_cap_usc.release()
        time.sleep(5)
        print("\n" + video_name + r" released" + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=r"./video_input")
    parser.add_argument("--out_folder", default=r"./video_output")
    parser.add_argument("--GPU", type=int, default=0)

    parser_args = parser.parse_args()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices)!=0:
        tf.config.experimental.set_memory_growth(physical_devices[parser_args.GPU], True)
        tf.config.experimental.set_visible_devices(physical_devices[parser_args.GPU], 'GPU')

    tf.keras.backend.clear_session()

    process_video(parser_args.input_folder, parser_args.out_folder)
