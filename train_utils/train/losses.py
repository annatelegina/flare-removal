import tensorflow as tf
import numpy as np

def advanced_gradient_loss(output_layer,output_true ):
    
     
    preds_gs = tf.image.rgb_to_grayscale(output_layer)
    gts_gs = tf.image.rgb_to_grayscale( output_true)
    
    filter1 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    filter2 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    filterx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    filtery = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    
    
    filters = np.concatenate([[filter1], [filter2],[filterx],[filtery]]) 
    filters = np.expand_dims(filters, -1)  # shape: (2, 3, 3, 1)
    filters = filters.transpose(1, 2, 3, 0)  # shape: (3, 3, 1,3)
    
    preds_out = tf.nn.conv2d((preds_gs ),
                   filters,
                   strides=[1, 1, 1, 1],
                   padding='SAME')
    gts_out = tf.nn.conv2d((gts_gs ),
                   filters,
                   strides=[1, 1, 1, 1],
                   padding='SAME')
    
    
    edge = tf.reduce_mean(tf.abs(   tf.abs( preds_out) - tf.abs(gts_out)    )   )
    

    return edge


def gradient_loss(output_layer,output_true ):
    
    print(222,333,output_layer.shape)
    preds_dy, preds_dx = tf.image.image_gradients( tf.image.rgb_to_grayscale(output_layer))
    gts_dy, gts_dx = tf.image.image_gradients( tf.image.rgb_to_grayscale( output_true))

    edge = tf.reduce_mean(tf.abs(   tf.abs(preds_dy) - tf.abs(gts_dy)    )   )+ tf.reduce_mean(tf.abs(   tf.abs(preds_dx) - tf.abs(gts_dx)    )   )
    return edge


def L1_loss(target, output):
    l1_loss_value = tf.reduce_mean(tf.abs(target - output))
    return l1_loss_value


def L2_loss(target, output):
    l2_loss_value = tf.reduce_mean(tf.square(target - output))
    return l2_loss_value


def L1_loss_vector(target, output):
    l1_loss_value = tf.reduce_mean(tf.abs(target - output), axis=[1, 2])
    return l1_loss_value


def L2_loss_vector(target, output):
    l2_loss_value = tf.reduce_mean(tf.square(target - output), axis=[1, 2])
    return l2_loss_value


def MS_SSIM_loss(target, output):
    ms_ssim_loss_value = (1 - tf.math.reduce_mean(tf.image.ssim_multiscale(target, output, 1)))
    return ms_ssim_loss_value


def make_VGG16_loss(blocks_dict, weights_path, loss_type='MAE'):
    """
    Creates VGG loss with given blocks of VGG16 with corresponding weights and loss type.
    :param blocks_dict: dict{block_num->int:block_weight->float}, pairs of required blocks and corresponding weights
    :param weights_path: str, path to VGG16 weights (notop)
    :param loss_type: str,"MAE" or "MSE"
    :return: function, VGG16_loss(target, output)
    """
    assert set(blocks_dict.keys()) <= {1, 2, 3, 4, 5}
    assert (loss_type=="MAE" or loss_type=="MSE")
    vgg16 = tf.keras.applications.VGG16(weights=None, include_top=False)
    vgg16.load_weights(weights_path)

    models = []
    for key in sorted(blocks_dict.keys()):
        if key in [1, 2]:
            models.append(tf.keras.Model(inputs=vgg16.input,
                                 outputs=vgg16.get_layer("block{}_conv2".format(key)).output))
        else:
            models.append(tf.keras.Model(inputs=vgg16.input,
                                         outputs=vgg16.get_layer("block{}_conv3".format(key)).output))

    def VGG16_loss(target, output):
        target = tf.keras.applications.vgg16.preprocess_input(target*255)
        output = tf.keras.applications.vgg16.preprocess_input(output*255)
        if loss_type=="MAE":
            VGG16_loss_array = [L1_loss(models[i](target), models[i](output)) *
                                blocks_dict[list(sorted(blocks_dict.keys()))[i]] for i in range(len(models))]
        else:
            VGG16_loss_array = [L2_loss(models[i](target), models[i](output)) *
                                blocks_dict[list(sorted(blocks_dict.keys()))[i]] for i in range(len(models))]
        vgg16_loss_value = tf.add_n(VGG16_loss_array)
        return vgg16_loss_value

    return VGG16_loss

def _get_weigthed_diff(diff_vec, gamma):
    eps = 1e-5
    diff_vec_new = tf.stack([vec * ((tf.reduce_max(vec)/(vec+eps)) ** gamma) for vec in tf.unstack(diff_vec)])        
    return diff_vec_new


def make_VGG16_loss_gamma(blocks_dict, weights_path, loss_type='MAE', gamma=0):
    """
    Creates VGG loss with given blocks of VGG16 with corresponding weights and loss type.
    :param blocks_dict: dict{block_num->int:block_weight->float}, pairs of required blocks and corresponding weights
    :param weights_path: str, path to VGG16 weights (notop)
    :param loss_type: str,"MAE" or "MSE"
    :return: function, VGG16_loss(target, output)
    """
    assert set(blocks_dict.keys()) <= {1, 2, 3, 4, 5}
    assert (loss_type == "MAE" or loss_type == "MSE")
    assert (0 <= gamma <= 1)

    vgg16 = tf.keras.applications.VGG16(weights=None, include_top=False)
    vgg16.load_weights(weights_path)

    models = []
    for key in sorted(blocks_dict.keys()):
        if key in [1, 2]:
            models.append(tf.keras.Model(inputs=vgg16.input,
                                 outputs=vgg16.get_layer("block{}_conv2".format(key)).output))
        else:
            models.append(tf.keras.Model(inputs=vgg16.input,
                                         outputs=vgg16.get_layer("block{}_conv3".format(key)).output))
    


    def VGG16_loss(target, output):
        target = tf.keras.applications.vgg16.preprocess_input(target*255)
        output = tf.keras.applications.vgg16.preprocess_input(output*255)
        if loss_type=="MAE":
            VGG16_loss_array = [tf.reduce_mean(_get_weigthed_diff(L1_loss_vector(models[i](target), models[i](output)), gamma=0.5))
                                * blocks_dict[list(sorted(blocks_dict.keys()))[i]] for i in range(len(models))]
        else:
            VGG16_loss_array = [tf.reduce_mean(_get_weigthed_diff(L2_loss_vector(models[i](target), models[i](output)), gamma=0.5))
                                * blocks_dict[list(sorted(blocks_dict.keys()))[i]] for i in range(len(models))]
        vgg16_loss_value = tf.add_n(VGG16_loss_array)
        return vgg16_loss_value

    return VGG16_loss

def make_DE76_loss(white_point='d65'):
    """
    Creates DE76 loss with given white point value
    :param white_point: white point from {"a", "b", "e", "d50", "d55", "d65", "icc"}
    :return: DE76 loss function
    """
    assert white_point in {"a", "b", "e", "d50", "d55", "d65", "icc"}
    _RGB_TO_XYZ = {
        "srgb": tf.constant([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]], tf.float32),
        }

    WHITE_POINTS = {item[0]: tf.reshape(tf.constant(item[1:]), [1,1,1,3]) for item in [
        ("a", 1.0985, 1.0000, 0.3558),
        ("b", 0.9807, 1.0000, 1.1822),
        ("e", 1.0000, 1.0000, 1.0000),
        ("d50", 0.9642, 1.0000, 0.8251),
        ("d55", 0.9568, 1.0000, 0.9214),
        ("d65", 0.9504, 1.0000, 1.0888),
        ("icc", 0.9642, 1.0000, 0.8249)
    ]}

    _XYZ_TO_LAB = tf.constant([[0.0, 116.0, 0.], [500.0, -500.0, 0.], [0.0, 200.0, -200.0]])
    _LAB_OFF = tf.reshape(tf.constant([16.0, 0.0, 0.0]), [1, 1, 1, 3])

    def _mul(coeffs, image):
        return tf.einsum("dc,bijc->bijd", coeffs, image)


    def rgb2xyz(rgb, space='srgb'):
        mask = rgb > 0.04045
        rgb1 = tf.pow((rgb + 0.055) / 1.055, 2.4)
        rgb2 = rgb / 12.92
        rgb = tf.where(mask, rgb1, rgb2)
        return _mul(_RGB_TO_XYZ[space], rgb)

    def _lab_f(x):
        x1 = tf.pow(x, 1.0 / 3.0)
        x2 = 7.787 * x + (16.0 / 116.0)
        return tf.where(x > 0.008856, x1, x2)

    def xyz2lab(xyz, white_point=white_point):
        xyz = xyz / WHITE_POINTS[white_point]
        f_xyz = _lab_f(xyz)
        return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF

    def rgb2lab(rgb, white_point="d65", space="srgb"):
        lab = xyz2lab(rgb2xyz(rgb, space), white_point)
        return lab

    def mean_deltaE_76(lab1, lab2):
        return tf.reduce_mean(tf.reduce_sum((lab1 - lab2) ** 2, axis=3) ** 0.5)

    def DE76_loss(target, output):
        gts_lab = rgb2lab(target)
        outputs_lab = rgb2lab(output)
        loss = mean_deltaE_76(gts_lab, outputs_lab)
        return loss

    return DE76_loss



