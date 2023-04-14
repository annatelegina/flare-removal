import sys
import numpy as np
sys.path.insert(0, "/home/p00536919/colour-demosaicing-develop")
from abc import ABC, abstractmethod

from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

from matplotlib import pyplot as plt



def get_raw_image(raw_img_path, height=2448, width=3264, depth=1023):
    with open(raw_img_path) as buf:
        img = np.fromfile(buf, dtype=np.uint16, count=height * width)
    img_reshaped = img.reshape((height, width))
    img_reshaped = img_reshaped[:, :, np.newaxis]
    img_reshaped = img_reshaped / depth
    return img_reshaped


def get_demosaiced_image(raw_img_path, height=2448, width=3264, depth=1023):
    with open(raw_img_path) as buf:
        img = np.fromfile(buf, dtype=np.uint16, count=height * width)
    img_reshaped = img.reshape((height, width))
    img_reshaped = img_reshaped / depth
    image_demo = demosaicing_CFA_Bayer_bilinear(img_reshaped, pattern='RGGB')
    image_demo = np.clip(image_demo, 0, 1)
    return image_demo


def get_raw_rggb_image(raw_img_path, height=2448, width=3264, depth=1023):
    with open(raw_img_path) as buf:
        img = np.fromfile(buf, dtype=np.uint16, count=height * width)
        #print(str(len(img))+"   "+raw_img_path) 
    img_reshaped = img[:].reshape((height, width))
    img_reshaped = img_reshaped[:, :, np.newaxis]
    img_reshaped = img_reshaped / depth
    raw_rgb_image = np.zeros((height // 2, width // 2, 4))
    raw_rgb_image[:, :, 0] = img_reshaped[::2, ::2, 0]
    raw_rgb_image[:, :, 1] = img_reshaped[::2, 1::2, 0]
    raw_rgb_image[:, :, 2] = img_reshaped[1::2, ::2, 0]
    raw_rgb_image[:, :, 3] = img_reshaped[1::2, 1::2, 0]
    return raw_rgb_image


def get_rgb_image(png_image_path):
    rgb_image_array = plt.imread(png_image_path)[:, :, :3]
    if np.max(rgb_image_array)>1:
        rgb_image_array = rgb_image_array / 255.0
    return rgb_image_array


def get_uint16_image(path, height=2448, width=3264, channels=3):
    with open(path, "rb") as f:
        buff = np.fromfile(f, count=height * width * channels, dtype=np.uint16)
    buff_rgb = buff.reshape(height, channels * width)
    buff_rgb_3 = buff_rgb.reshape(height, width, channels)[:, :, ::-1] / 4095
    return buff_rgb_3


class ImageReader(ABC):
    """
    Class for reading an arbitrary image and performing transformation chain on it
    """

    def __init__(self, transform_list=None):
        self.transform_list = transform_list

    @abstractmethod
    def _get_image(self, path):
        pass

    def get_transformed_image(self, path):
        image = self._get_image(path)
        if self.transform_list is not None:
            if len(self.transform_list)!=0:
                for transform in self.transform_list:
                    image = transform.transform(image)
        return image


class ImageReaderDemosaiced(ImageReader):

    def __init__(self, height=2448, width=3264, depth=1023, transform_list=None):
        super().__init__(transform_list)
        self.height = height
        self.width = width
        self.depth = depth

    def _get_image(self, path):
        image = get_demosaiced_image(path, height=self.height, width=self.width, depth=self.depth)
        return image


class ImageReaderBayer(ImageReader):

    def __init__(self, height=2448, width=3264, depth=1023, transform_list=None):
        super().__init__(transform_list)
        self.height = height
        self.width = width
        self.depth = depth

    def _get_image(self, path):
        image = get_raw_image(path, height=self.height, width=self.width, depth=self.depth)
        return image
    
class ImageReaderUINT16(ImageReader):

    def __init__(self, height=2448, width=3264, channels=3, transform_list=None):
        super().__init__(transform_list)
        self.height = height
        self.width = width
        self.channels = channels

    def _get_image(self, path):
        image = get_uint16_image(path, height=self.height, width=self.width, channels=self.channels)
        return image


class ImageReaderRGGB(ImageReader):

    def __init__(self, height=2448, width=3264, depth=1023, transform_list=None):
        super().__init__(transform_list)
        self.height = height
        self.width = width
        self.depth = depth

    def _get_image(self, path):
        image = get_raw_rggb_image(path, height=self.height, width=self.width, depth=self.depth)
        return image


class ImageReaderPNG(ImageReader):
    
    def __init__(self, transform_list=None):
        super().__init__(transform_list)

    def _get_image(self, path):
        image = get_rgb_image(path)
        return image
