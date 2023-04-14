import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras 


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


def downscale_generator(height=None, width=None, input_channels=3, filters=32):

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
    
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='downscale_net')
    return _model  


def full_res_generator(height=None, width=None, input_channels=6, filters=16):

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
    x = tf.keras.layers.Subtract()([inputs[:,:3,:,:], x])
    
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='full_res_net')
    return _model       

def get_derain_net(height=None, width=None, input_channels=3, filters=32):

    inputs = tf.keras.Input(shape=[input_channels, height, width])

    x = conv_block(growth_rate=2, filters=filters, kernel_size=(5, 5), strides=1, x=inputs)
    res1 = x

    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=2, x=x)
    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    res2 = x

    x = conv_block(growth_rate=4, filters=filters, kernel_size=(3, 3), strides=2, x=x)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)

    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=2, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=4, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=8, x=x)
    x = dilated_conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), dilation_rate=16, x=x)

    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)

    x = deconv_block(growth_rate=4, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res2])
    x = conv_skip_block(growth_rate=4, filters=filters, kernel_size=(1, 1), x=x)

    x = conv_block(growth_rate=8, filters=filters, kernel_size=(3, 3), strides=1, x=x)

    x = deconv_block(growth_rate=2, filters=filters, kernel_size=(4, 4), strides=2, x=x)

    x = tf.keras.layers.Concatenate(axis=1)([x, res1])
    x = conv_skip_block(growth_rate=2, filters=filters, kernel_size=(1, 1), x=x)

    x = conv_block(growth_rate=1, filters=filters, kernel_size=(3, 3), strides=1, x=x)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', data_format='channels_first')(x)
    x = tf.keras.layers.Subtract()([inputs[:,:3,:,:], x])
    _model = tf.keras.Model(inputs=inputs, outputs=x, name='derain_net')
    return _model 