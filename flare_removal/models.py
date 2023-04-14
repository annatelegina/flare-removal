import tensorflow as tf
from enum import Enum


class DataFormat(Enum):
    NCHW = 'channels_first'
    NHWC = 'channels_last'


class NetBlocksFactory:
    def __init__(self, data_format=DataFormat.NHWC):
        self.data_format = data_format

    def conv_block(self, growth_rate, filters, kernel_size, strides, x):
        x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', strides=strides,
                                   data_format=self.data_format.value)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def dilated_conv_block(self, growth_rate, filters, kernel_size, dilation_rate, x):
        x = tf.keras.layers.Conv2D(growth_rate * filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                                   data_format=self.data_format.value)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def conv_skip_block(self, growth_rate, filters, kernel_size, x):
        x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same',
                                            data_format=self.data_format.value)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def deconv_block(self, growth_rate, filters, kernel_size, strides, x):
        x = tf.keras.layers.Conv2DTranspose(growth_rate * filters, kernel_size, padding='same', strides=strides,
                                            data_format=self.data_format.value)(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), 1, padding='same', data_format=self.data_format.value)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x
    
    def downsample(self, filters, kernel_size, apply_batchnorm, act, x):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides = (2, 2), padding = 'same', use_bias = False)(x)
        if apply_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)

        if act:
            x = tf.keras.layers.LeakyReLU()(x)

        return x


def get_vanilla_downscale_net(height=None, width=None, input_channels=3, filters=32, data_format=DataFormat.NHWC):
    blocks = NetBlocksFactory(data_format)

    if data_format == DataFormat.NHWC:
        inputs = tf.keras.Input(shape=[height, width, input_channels])
        concat_axis = -1
    elif data_format == DataFormat.NCHW:
        inputs = tf.keras.Input(shape=[input_channels, height, width])
        concat_axis = 1

    x = blocks.dilated_conv_block(growth_rate=2, filters=filters, kernel_size=(9, 9), dilation_rate=16, x=inputs)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(5, 5), dilation_rate=8, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res1 = x

    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res2 = x

    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)

    x = blocks.deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res2])

    x = blocks.conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = blocks.deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res1])

    x = blocks.conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), dilation_rate=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format=data_format.value)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.Subtract()([inputs, x])
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', data_format=data_format.value)(x)

    _model = tf.keras.Model(inputs=inputs, outputs=x, name='downscale_net')
    return _model


def get_vanilla_fullres_net(height=None, width=None, input_channels=6, filters=16, data_format=DataFormat.NHWC):

    blocks = NetBlocksFactory(data_format)

    if data_format == DataFormat.NHWC:
        inputs = tf.keras.Input(shape=[height, width, input_channels])
        concat_axis = -1
    elif data_format == DataFormat.NCHW:
        inputs = tf.keras.Input(shape=[input_channels, height, width])
        concat_axis = 1

    x = blocks.dilated_conv_block(growth_rate=2, filters=filters, kernel_size=(5, 5), dilation_rate=8, x=inputs)
    res1 = x

    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res2 = x

    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)

    x = blocks.deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res2])

    x = blocks.conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = blocks.deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res1])

    x = blocks.conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), dilation_rate=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format=data_format.value)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    if data_format == DataFormat.NHWC:
        x = tf.keras.layers.Subtract()([inputs[:, :, :, :3], x])
    elif data_format == DataFormat.NCHW:
        x = tf.keras.layers.Subtract()([inputs[:, :3, :, :], x])
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='fullres_net')
    return _model


#------------------------------------MODEL WITH ALPHA-------------------------------------------------------------------

def vanilla_downscale_net_alpha(height=None, width=None, input_channels=3, filters=32, data_format=DataFormat.NHWC):
    
    blocks = NetBlocksFactory(data_format)
    
    alpha = tf.keras.Input(shape=[1, 1, 1])
    
    if data_format == DataFormat.NHWC:
        inputs = tf.keras.Input(shape=[height, width, input_channels])
        concat_axis = -1
    elif data_format == DataFormat.NCHW:
        inputs = tf.keras.Input(shape=[input_channels, height, width])
        concat_axis = 1

    x = blocks.dilated_conv_block(growth_rate=2, filters=filters, kernel_size=(9, 9), dilation_rate=16, x=inputs)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(5, 5), dilation_rate=8, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res1 = x
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    
    sigmas1 = blocks.conv_block(1, 128,  (1, 1),(1, 1), x=alpha)
    x = x * sigmas1
    
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    res2 = x
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    
    sigmas2 = blocks.conv_block(1, 128,  (1, 1), (1, 1), x=alpha)
    x = x * sigmas2
    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    
    sigmas3 = blocks.conv_block(1, 256,  (1, 1), (1, 1), x=alpha)
    x = x * sigmas3
    
    x = blocks.deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res2])

    x = blocks.conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = blocks.deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res1])

    x = blocks.conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)
    x = blocks.dilated_conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), dilation_rate=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format=data_format.value)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.Subtract()([inputs, x])
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', data_format=data_format.value)(x)

    _model = tf.keras.Model(inputs=[inputs, alpha], outputs=x, name='downscale_net_alpha')
    return _model


#------------------------------------ DISCRIMINATOR -------------------------------------------------------------------

def get_patch_gan(input_channels=3, filters=64, data_format=DataFormat.NHWC):
    blocks = NetBlocksFactory(data_format)

    if data_format == DataFormat.NHWC:
        inputs = tf.keras.Input(shape=[None, None, input_channels], name='input_image')
        targets = tf.keras.Input(shape=[None, None, input_channels], name='target_image')
    elif data_format == DataFormat.NCHW:
        inputs = tf.keras.Input(shape=[input_channels, None, None], name='input_image')
        targets = tf.keras.Input(shape=[input_channels, None, None], name='target_image')

    x = tf.keras.layers.concatenate([inputs, targets])  # (batch_size, 256, 256, channels*2)

    x = blocks.downsample(filters, 4, False, True, x=x)
    x = blocks.downsample(filters * 2, 4, True, True, x=x)
    x = blocks.downsample(filters * 4, 4, True, True, x=x)
    x = blocks.downsample(filters * 8, 4, True, True, x=x)
    x = blocks.conv_block(growth_rate=1, filters=1, kernel_size=(1, 1), strides=(1, 1), x=x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    _model = tf.keras.Model(inputs=[inputs, targets], outputs=x, name='patch_gan')

    return _model


#------------------------------------ FAST FULL RESOLUTION MODEL VARIANTS -------------------------------------------------------------------

def get_tiny_fullres_net(height=None, width=None, input_channels=6, filters=16, data_format=DataFormat.NHWC):
    blocks = NetBlocksFactory(data_format)

    if data_format == DataFormat.NHWC:
        inputs = tf.keras.Input(shape=[height, width, input_channels])
        concat_axis = -1
    elif data_format == DataFormat.NCHW:
        inputs = tf.keras.Input(shape=[input_channels, height, width])
        concat_axis = 1

    x = blocks.conv_block(growth_rate=1, filters=24, kernel_size=(2, 2), strides=(2, 2), x=inputs)
    x = blocks.conv_block(growth_rate=2, filters=filters, kernel_size=(3, 3), strides=(1, 1), x=x)
    res1 = x

    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.conv_block(growth_rate=2, filters=filters, kernel_size=(3, 3), strides=(1, 1), x=x)
    res2 = x

    x = tf.keras.layers.AveragePooling2D((2, 2), 2, padding='same', data_format=data_format.value)(x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = blocks.dilated_conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)

    x = blocks.deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=(2, 2), x=x)
    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res2])
    x = blocks.conv_block(growth_rate=2, filters=filters, kernel_size=(3, 3), strides=(1, 1), x=x)

    x = blocks.deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=(2, 2), x=x)
    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, res1])
    x = blocks.conv_block(growth_rate=2, filters=filters, kernel_size=(3, 3), strides=(1, 1), x=x)

    x = tf.keras.layers.Conv2D(12, (3, 3), padding='same', data_format=data_format.value)(x)
    x = blocks.deconv_block(growth_rate=1, filters=3, kernel_size=(2, 2), strides=(2, 2), x=x)
    x = tf.keras.layers.Activation('tanh')(x)

    if data_format == DataFormat.NHWC:
        x = tf.keras.layers.Subtract()([inputs[:, :, :, :3], x])
    elif data_format == DataFormat.NCHW:
        x = tf.keras.layers.Subtract()([inputs[:, :3, :, :], x])
    
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='fullres_net')

    return _model