import tensorflow as tf
import logging

from typing import List

logging.basicConfig(level=logging.INFO)


class CxLoss(tf.keras.losses.Loss):
    def __init__(self, distance_type="l2", use_crop=False, max_sampling_size=100, sigma=1.0, beta=1.0):
        super().__init__()

        self.beta = beta
        self.sigma = sigma
        self.use_crop = use_crop

        backbone = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

        layers = [f"block{idx}_conv2" for idx in range(1, 4)]

        outputs = [backbone.get_layer(name).output for name in layers]

        self.model = tf.keras.Model(inputs=backbone.input, outputs=outputs)

        self.max_sampling_size = max_sampling_size

        self.dist = None

        if distance_type == "l2":
            self.dist = self._l2_dist
        else:
            self.dist = self._dot_dist

    def _l2_dist(self, inps, target):
        axis_c = 3
        axis_n = 0

        inps_shape = inps.shape.as_list()
        target_shape = target.shape.as_list()

        inps_vecs = tf.reshape(inps, (inps_shape[axis_n], -1, inps_shape[axis_c]))
        target_vecs = tf.reshape(target, (target_shape[axis_n], -1, target_shape[axis_c]))

        r_inps_s = tf.reduce_sum(inps_vecs * inps_vecs, axis=2)
        r_target_s = tf.reduce_sum(target_vecs * target_vecs, axis=2)

        raw_distances_list = []

        for i in range(target_shape[axis_n]):
            iv, tv = inps_vecs[i], target_vecs[i]
            ri, rt = r_inps_s[i], r_target_s[i]

            A = tf.matmul(tv, iv, transpose_b=True)

            rt = tf.reshape(rt, [-1, 1])

            dist = rt - 2 * A + ri

            cs_shape = inps_shape[:3] + [dist.shape[0]]

            cs_shape[0] = 1

            dist = tf.reshape(tf.transpose(dist), cs_shape)

            dist = tf.maximum(float(0.0), dist)

            raw_distances_list.extend([dist])

        raw_distances = tf.convert_to_tensor(
            [tf.squeeze(rd, axis=0) for rd in raw_distances_list]
        )

        relative_dist = compute_relative_dist(raw_distances, axis=axis_c)
        cx = CxLoss.compute_cx(relative_dist, beta=self.beta, sigma=self.sigma, axis=axis_c)

        return cx

    def _dot_dist(self, inps, target):
        pass

    @staticmethod
    def compute_cx(dist_normalized, beta, sigma, axis=3):
        w = tf.exp((beta - dist_normalized) / sigma)

        cx = w / tf.reduce_sum(w, axis=axis, keepdims=True)

        return cx

    def _get_features(self, inps):
        with tf.device("/GPU:0"):
            feats = self.model(inps, training=False)

            if not isinstance(feats, list):
                # feats = tf.expand_dims(feats, 0)
                feats = [feats]

        return feats

    def call(self, predict, target):
        predict = self._get_features(predict)
        target = self._get_features(target)

        dists = self._compute_cx_loss(predict, target)

        hw_axis = (1, 2)

        losses = []

        for idx, cs in enumerate(dists):
            k_max = tf.reduce_max(cs, axis=hw_axis)

            cs = tf.reduce_mean(k_max, axis=1)

            cx_loss_val = 1.0 - cs

            cs_loss_val = -tf.math.log(1.0 - cx_loss_val)

            cx_loss_val = tf.reduce_mean(cs_loss_val)

            # correspond to conv3_block2
            if idx == len(dists)-1:
                cx_loss_val *= 0.5

            losses.append(cx_loss_val)

        return tf.reduce_mean(losses)

    def _compute_cx_loss(self, predict, target) -> List:
        dists = []

        for p, t in zip(predict, target):
            if self.use_crop:
                p = crop_quarters(p)
                t = crop_quarters(t)

            N, fH, fW, fC = p.shape.as_list()

            if fH * fW <= self.max_sampling_size ** 2:
                logging.warning("Skipping pooling for CX....")
            else:
                logging.info(f"pooling for CX {self.max_sampling_size**2} out of {fH*fW}")
                p, t = random_pooling([p, t], output_1d_size=self.max_sampling_size)

            cs = self.dist(p, t)

            dists.append(cs)

        return dists


class CoBiLoss(CxLoss):
    def __init__(self, distance_type="l2", use_crop=False, max_sampling_size=100, sigma=1.0, beta=1.0):
        super().__init__(distance_type, use_crop, max_sampling_size, sigma, beta)

    def _compute_cobi_loss(self, predict, target):
        hw_axis = (1, 2)
        weight_spatial = 0.1

        cs_spatial = []

        for t in target:
            grid = get_meshgrid(t)

            cs_spatial.append(self._l2_dist(grid, grid))

        losses = []

        for p, t, csp in zip(predict, target, cs_spatial):
            cs_features = self.dist(p, t)

            if csp.shape != cs_features.shape:
                raise RuntimeError(f"Invalid shape, got {cs_spatial.shape}, {cs_features.shape}")

            for idx in range(csp.shape[0]):
                cs_sp = tf.expand_dims(csp[idx], axis=0)
                cs_f = tf.expand_dims(cs_features[idx], axis=0)

                cs_combined = cs_f * (1.0 - weight_spatial) + cs_sp * weight_spatial

                k_max = tf.reduce_max(cs_combined, axis=hw_axis)

                cs = tf.reduce_mean(k_max, axis=1)

                cx_loss_val = 1.0 - cs

                cs_loss_val = -tf.math.log(1.0 - cx_loss_val)

                cx_loss_val = tf.reduce_mean(cs_loss_val)

                losses.append(cx_loss_val)

        return tf.reduce_mean(losses)

    def call(self, predict, target):
        s = 5
        r = 1

        predict_rgb = tf.image.extract_patches(
            predict,
            sizes=[1, s, s, 1], strides=[1, 1, 1, 1],
            rates=[1, r, r, 1], padding="same".upper()
        )

        predict_rgb = [predict_rgb]

        target_rgb = tf.image.extract_patches(
            target,
            sizes=[1, s, s, 1], strides=[1, 1, 1, 1],
            rates=[1, r, r, 1], padding="same".upper()
        )

        target_rgb = [target_rgb]

        rgb_loss = self._compute_cobi_loss(predict_rgb, target_rgb)

        predict_feats = self._get_features(predict)
        target_feats = self._get_features(target)

        features_loss = self._compute_cobi_loss(predict_feats, target_feats)

        return rgb_loss + features_loss


def get_meshgrid(features):
    if len(features.shape) != 4:
        features = tf.expand_dims(features, 0)

    N, H, W, C = features.shape.as_list()

    rows = tf.range(H)
    cols = tf.range(W)

    rows = tf.cast(rows, dtype=tf.float32)/(H)
    cols = tf.cast(cols, dtype=tf.float32)/(W)

    grid = tf.meshgrid(rows, cols)

    grid = tf.concat(
        [tf.expand_dims(gi, 2) for gi in grid], axis=2
    )

    grid = tf.expand_dims(grid, axis=0)

    grid = tf.tile(grid, multiples=tf.constant([N, 1, 1, 1]))

    return grid


def random_sampling(tensor_NHWC, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_NHWC).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_NHWC, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random.shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices

    res = tf.gather(tensor_NSC, indices, axis=1)

    return res, indices


def random_pooling(feats, output_1d_size=100):
    is_input_tensor = type(feats) is tf.Tensor

    if is_input_tensor:
        feats = [feats]

    # convert all inputs to tensors
    feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, H, W, C = feats[0].shape.as_list()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res


def crop_quarters(features):
    N, fH, fW, fC = features.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(features, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(features, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(features, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(features, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)

    return feature_tensor


def l2_normalize_cw(features):
    axis_c = 3

    norms = tf.norm(features, ord='euclidean', axis=axis_c)

    norms_expanded = tf.expand_dims(norms, axis_c)
    features = tf.divide(features, norms_expanded)

    return features


def compute_relative_dist(dist_raw, axis=3):
    dist_min = tf.reduce_min(dist_raw, axis=axis, keepdims=True)

    dist_normalized = dist_raw / (dist_min + 1e-5)

    return dist_normalized


def center_by_t(I, T):
    axis = (0, 1, 2)

    mean = tf.reduce_mean(T, axis)

    I_centered = I - mean
    T_centered = T - mean

    return I_centered, T_centered


def patch_decomposition(T):
    patch_size = 1
    patches_as_depth_vectors = tf.extract_image_patches(
        images=T, ksizes=[1, patch_size, patch_size, 1],
        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="valid"
    )

    patches_NHWC = tf.reshape(
        patches_as_depth_vectors,
        shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3]],
    )

    patches_HWCN = tf.transpose(
        patches_NHWC,
        perm=[1, 2, 3, 0],
    )

    return patches_HWCN


cx_loss = None
cobi_loss = None


def get_context_loss():
    global cx_loss

    if cx_loss is None:
        cx_loss = CxLoss(sigma=0.5, beta=1.0, max_sampling_size=63)

    return cx_loss


def get_cobi_loss():
    global cobi_loss

    if cobi_loss is None:
        cobi_loss = CoBiLoss(sigma=0.5, beta=1.0, max_sampling_size=63)

    return cobi_loss
