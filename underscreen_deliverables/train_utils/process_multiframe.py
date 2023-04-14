
import skimage
import math
import cv2
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import matplotlib

# from utils import image_io, dataset
import numpy as np
import skimage
import cv2
import math









def pixel_shuffle_in(raw, cfa_size=2):
    """
    Transform CFA pattern into RGGB tensor
    Input
    ----------
    raw: CFA Bayer pattern(RGGB) with shape [H, W]
    return

    Returns
    ----------
    RGGB with shape [4, H/2, W/2]

    """
    h, w = raw.shape
    data = np.empty((cfa_size ** 2, h // cfa_size, w // cfa_size), raw.dtype)
    i = 0
    for y in range(cfa_size):
        for x in range(cfa_size):
            data[i, :, :] = raw[y::cfa_size, x::cfa_size]
            i += 1
    return data

def wb_gain_raw_f_one(raw_f_ps, wb_gains=[2.3, 1.0, 2.7]):
    r_gain = wb_gains[0]
    g_gain = wb_gains[1]
    b_gain = wb_gains[2]
    raw_out=np.zeros_like(raw_f_ps)

    r =  raw_f_ps[0,:,:]
    gr = raw_f_ps[1,:,:]
    gb = raw_f_ps[2,:,:]
    b =  raw_f_ps[3,:,:]

    raw_out[0,:,:] = r*r_gain
    raw_out[1,:,:] = gr*g_gain
    raw_out[2,:,:] = gb*g_gain
    raw_out[3,:,:] = b*b_gain

    raw_out = np.clip(raw_out,0, 1)

    return raw_out
def rggb_to_rgb(burst_rggb, isAWB = False):
    """
    raw with shape burst_rggb  [N*4, H, W] -> [N, H, W, 3]
    """
    rgb_imgs = []
    for i in range(len(burst_rggb)//4):
        if isAWB:
            rggb = wb_gain_raw_f_one(burst_rggb[i*4: i*4+4])
        else:
            rggb = burst_rggb[i*4: i*4+4]
        r =  rggb[0, :,:]
        g = (rggb[1, :,:] + rggb[2, :,:])/2.0
        b =  rggb[3, :,:]
        rgb = np.stack((r, g, b), axis=-1)
        rgb_imgs.append(rgb)
    return np.stack(rgb_imgs).squeeze()
def depth_to_space_1(patch, size=2):
    """
    Transform RGGB tensor into CFA pattern
    Input
    ----------
    patch: RGGB with shape [4, H, W]

    Returns
    ----------
    raw: CFA with shape [2*H, 2*W]
    """
    _, h, w = patch.shape
    newwidth, newheight = w * size, h * size
    data = np.zeros((newheight, newwidth), dtype=np.float64)

    chn_index = 0
    for yy in range(size):
        for xx in range(size):
            data[yy::size, xx::size] = patch[chn_index,:, :]
            chn_index += 1
    return data



def rggb_to_burst(burst_rggb):
    """
    raw with shape [N*4, H/2, W/2] [N, H, W] ->
    """
    burst = []
    for i in range(len(burst_rggb)//4):
        cfa = depth_to_space_1(burst_rggb[i*4: i*4+4])
        burst.append(cfa)
    return np.stack(burst, axis = 0)

def burst_to_rggb(raw):
    """
    raw with shape [N, H, W] -> [N*4, H/2, W/2]
    """
    rggb_imgs = []
    for frame in raw:
        rggb = pixel_shuffle_in(frame)
        rggb_imgs.append(rggb)
    return np.concatenate(rggb_imgs)


def ghost_estimate_new(img1, img2, thresh=10, down_sample=4, idx=0, isDebug=False):
    """
    Input is img1 and img2 in range [0, 1]
    Return mask in range [0, 1]
    """
    src_h1, src_w1, c1 = img1.shape
    dst_size = (src_w1//down_sample, src_h1//down_sample)

    #gamma=5
    #img1 = img1**(1/gamma)
    #img2 = img2**(1/gamma)
    # bicubic in downsample (4)  times smaller
    img1 = (img1 * 255).astype('uint8')
    img2 = (img2 * 255).astype('uint8')
    
    
    frame1_NR = cv2.bilateralFilter(img1, 12, 20, 75)/255
    frame2_aligned_NR = cv2.bilateralFilter(img2, 12, 20, 75)/255
    
    
    
    difference = np.abs(frame1_NR ** 0.2-frame2_aligned_NR ** 0.2)*(1/frame1_NR**2)

    threshold = 0.2

    ghost_map_1 = np.sum(np.abs(difference), axis=-1)/3>threshold
    ghost_map_1 = cv2.blur( (ghost_map_1*255).astype('uint8'),(7,7))/255
    ghost_map_1 =ghost_map_1>threshold
    mask_one_out= ghost_map_1*255
    mask_one_out = mask_one_out / 255.

    return mask_one_out







def ghost_estimate(img1, img2, thresh=10, down_sample=4, idx=0, isDebug=False):
    """
    Input is img1 and img2 in range [0, 1]
    Return mask in range [0, 1]
    """
    src_h1, src_w1, c1 = img1.shape
    dst_size = (src_w1//down_sample, src_h1//down_sample)

    gamma=2.2
    img1 = img1**(1/gamma)
    img2 = img2**(1/gamma)
    # bicubic in downsample (4)  times smaller
    img1 = (img1 * 255).astype('uint8')
    img2 = (img2 * 255).astype('uint8')
    
    
    
    
    print(np.max(img1))
    img1_ds=cv2.resize(img1, dst_size, interpolation=3)
    img2_ds = cv2.resize(img2, dst_size,interpolation=3)
    print(np.max(img1_ds))

    img1_ds = img1_ds.astype(np.float32)
    img2_ds = img2_ds.astype(np.float32)

    mask_rgb = np.abs(img1_ds - img2_ds)
    mask_rgb = np.where(mask_rgb > thresh, 255, np.zeros_like(mask_rgb))

    mask_one = np.sum(mask_rgb, axis = -1)
    mask_one = np.where(mask_one>thresh, 255, 0)
    mask_one = mask_one.astype(np.uint8)

    # dilate just take a maximum from rectangular (5,5). It allows grab nearest points.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_one = cv2.dilate(mask_one, kernel)

    mask_one_out = cv2.resize(mask_one, (src_w1, src_h1))

    mask_one_out = np.where(mask_one_out>thresh, 255, 0)
    # also make blur MA moving average.
    mask_one_out = cv2.blur(mask_one_out, (5,5))

    mask_one_out = mask_one_out / 255.

    return mask_one_out
def get_ORB_aligner(MIN_MATCH_COUNT=10, nfeatures=100000, edgeThreshold=6, fastThreshold=3, is_awb=False):

    orb = cv2.ORB_create(nfeatures, edgeThreshold=edgeThreshold,  fastThreshold=fastThreshold)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6,
                       key_size = 12,
                       multi_probe_level = 1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    def aligner(ref, source):
        """
        RGGB format channels last datatype
        channels first: 4, H, W,
        ref_rgb in [0,1] range.

        """
        ref_rgb = rggb_to_rgb(ref, isAWB = is_awb)
        source_rgb = rggb_to_rgb(source, isAWB = is_awb)

        ref_rgb = (ref_rgb*255).clip(0,255).astype(np.uint8)
        source_rgb = (source_rgb*255).clip(0,255).astype(np.uint8)

        ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)

        keypoints_ref, descriptors1 = orb.detectAndCompute(ref_gray, None)
        keypoints_source, descriptors2 = orb.detectAndCompute(source_gray, None)
#         print(descriptors1.shape[0], descriptors2.shape[0] )
        if (descriptors2 is None or descriptors2.shape[0]<=2  ) or (descriptors1 is None or descriptors1.shape[0]<=2):
            return source


        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        goodMatch = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            # if m much closer to query than n we add it as good matcher
            if (m.distance < 0.7 * n.distance):
                goodMatch.append(m)


        if len(goodMatch) > MIN_MATCH_COUNT:
            ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            source_pts = np.float32([keypoints_source[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(ref_pts, source_pts, cv2.RANSAC, 5.0)
            warp_source = cv2.warpPerspective(source.transpose(1, 2, 0),  np.linalg.inv(M),
                                              (source.shape[2],  source.shape[1]),
                                             flags = cv2.INTER_NEAREST)
            warp_source = warp_source.transpose(2, 0, 1)
        else:
            warp_source = source

        return warp_source

    return aligner


def align_burst_rggb(burst_rggb, aligner):
    """
    Input is burst in RGGB format [burst_size * 4, H, W]
    Output is transformed burst with some align function
    aligner - with input( ref, source)
    some model to make alignment of two images. STN for example
    """
    # make alignment base on ref
    channels, H, W = burst_rggb.shape

    ref = burst_rggb[:4]
    sources = burst_rggb[4:]

    aligned_frames = burst_rggb[:4]
    masks = np.ones_like(ref[:1])
    fusion_frames = burst_rggb[:4]

    n_frames = len(burst_rggb)//4
    for i in range(n_frames-1):
        source = sources[i*4:i*4+4]
        warp_source = aligner(ref, source)
        aligned_frames = tf.concat([aligned_frames, warp_source], axis=0)

        mask = ghost_estimate_new(rggb_to_rgb(ref, False), rggb_to_rgb(warp_source, False),
                              thresh = 10, down_sample=4 )

        masks = tf.concat([masks, mask[tf.newaxis, :]], axis=0)
        fuse = mask * ref + (1-mask) * warp_source
        fusion_frames = tf.concat([fusion_frames, fuse], axis=0)

    return aligned_frames, masks, fusion_frames

def align_burst(burst):
    ORB_aligner = get_ORB_aligner()
    burst_rggb = burst_to_rggb(burst)

    _, masks, wraped_burst_rggb = align_burst_rggb(burst_rggb, ORB_aligner)

    wraped_burst_cfa = rggb_to_burst(wraped_burst_rggb)
    return wraped_burst_cfa,masks
