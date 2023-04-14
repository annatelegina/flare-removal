import argparse
import numpy as np
import cv2
import tensorflow as tf
from typing import List, Union, Iterable
from pathlib import Path
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt
from skimage.util import view_as_windows

"""
Dependencies:
    tensorflow
    opencv
    tqdm
"""

class InferenceProcessor:
    def __init__(self, pad_factor: int, downscale_factor: int):
        self._pad_factor = pad_factor
        self._downscale_factor = downscale_factor


    def pad_image(self, image):
        height, width = image.shape[:2]
        pad_0 = int((self._pad_factor - height % self._pad_factor) % self._pad_factor)
        pad_1 = int((self._pad_factor - width % self._pad_factor) % self._pad_factor)
        paddings = tf.constant([[0, 0], [0, pad_0], [0, pad_1], [0, 0]])
        padded_image = tf.cast(image[np.newaxis, :, :, :], tf.float32)
        padded_image = tf.pad(padded_image, paddings, "SYMMETRIC")
        pad_values = [pad_0, pad_1]
        return padded_image, pad_values

    def preprocess_downscale(self, image: np.array) -> np.array:
        if self._downscale_factor != 1:
            image = cv2.resize(image, (image.shape[1]//self._downscale_factor, image.shape[0]//self._downscale_factor),
                                interpolation=cv2.INTER_AREA)
        image = image/255
        image = np.clip(image, 0, 1)
        image, pad_values = self.pad_image(image)
        return image, pad_values

    def preprocess_fullres(self, image: np.array, guidance: np.array) -> np.array:
        image = image/255
        guidance = cv2.resize(guidance, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        guidance = np.clip(guidance, 0, 1)

        net_input = np.concatenate([image, guidance], axis=-1)
        net_input, pad_values = self.pad_image(net_input)
        return net_input, pad_values

    def postprocess(self, image: np.array, pad_values) -> np.array:
        pad_values = [-x if x != 0 else None for x in pad_values]
        image = image[0, :pad_values[0], :pad_values[1]]
        return image

class BasicInfrenceProvider:
    def __init__(self, model):
        self._model = model
    
    def __call__(self, image):
        """
        Parameters:
            image: preprocessed input for iference model
        """
        return self._model.predict([image])


class PatchInferenceProvider(BasicInfrenceProvider):
    def __init__(self, model, patch_size, overlap_size):
        super().__init__(model)
        self._patch_size = patch_size
        self._overlap_size = overlap_size
    
    def __call__(self, image):
        """
        Parameters:
            image: preprocessed input for iference model
        """
        image = image.numpy()[0]
        patches = self._form_patches(image)
        patches = np.array(patches)
        res = self._model.predict(patches)

        res = self._build_result(res, image.shape)
        return res[np.newaxis, ...]

    def _build_result(self, pred_patches, image_shape):
        result = np.zeros((*image_shape[:-1], pred_patches[0].shape[-1]), dtype=pred_patches[0].dtype)
        strides = (self._patch_size[0] - self._overlap_size[0], self._patch_size[1] - self._overlap_size[1])
        row_patch_len = image_shape[1] // strides[1]

        for i in range(len(pred_patches) // row_patch_len) :
            for j in range(row_patch_len):
                idx = i*row_patch_len + j
                patch = pred_patches[idx]
                patch = self._preprocess_patch_borders(patch, i, j, len(pred_patches) // row_patch_len, row_patch_len)
                result[i*strides[0]: i*strides[0] + self._patch_size[0], j*strides[1]: j*strides[1] + self._patch_size[1]] += patch
        return result

    def _preprocess_patch_borders(self, patch, i, j, i_end, j_end):
        if i != 0:
            coefs = np.arange( 0.1, 0.9, (0.9-0.1)/self._overlap_size[0])
            patch[:self._overlap_size[0]] =  (patch[:self._overlap_size[0]].transpose(1, 2, 0) * coefs).transpose(2, 0, 1)
        if i != i_end:
            coefs = np.arange( 0.9, 0.1, -(0.9-0.1)/self._overlap_size[0])
            patch[-self._overlap_size[0]:] = (patch[-self._overlap_size[0]:].transpose(1, 2, 0) * coefs).transpose(2, 0, 1)
        if j != 0:
            coefs = np.arange( 0.1, 0.9, (0.9-0.1)/self._overlap_size[1])
            patch[:, :self._overlap_size[1]] =  (patch[:, :self._overlap_size[1]].transpose(0, 2, 1) * coefs).transpose(0, 2, 1)
        if j != j_end:
            coefs = np.arange( 0.9, 0.1, -(0.9-0.1)/self._overlap_size[1])
            patch[:, -self._overlap_size[1]:] =  (patch[:, -self._overlap_size[1]:].transpose(0, 2, 1) * coefs).transpose(0, 2, 1)
        return patch
    
    def _form_patches(self, image):
        patches = view_as_windows(image, (*self._patch_size, image.shape[-1]))
        ret = []
        strides = (self._patch_size[0] - self._overlap_size[0], self._patch_size[1] - self._overlap_size[1])
        for i in range(patches.shape[0] // strides[0] + 1):
            for j in range(patches.shape[1] // strides[1] + 1):
                ret.append(patches[i*strides[0], j*strides[1], 0])
        return ret


def infer_downscale_model(generator, input_imgs:List[Union[str, Path]], 
                            processor: InferenceProcessor) -> Iterable[np.array]:
    for image_path in input_imgs:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # load image and covert BGR -> RGB
        image, pad_values = processor.preprocess_downscale(image)
        
        result = generator(image)
        result = processor.postprocess(result, pad_values)
        yield result


def infer_fullres_model(generator, input_imgs:List[Union[str, Path]], guidances: List[np.array], 
                        processor: InferenceProcessor) -> Iterable[np.array]:
    for image_path, guidance in zip(input_imgs, guidances):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # load image and covert BGR -> RGB
        
        net_input, pad_values = processor.preprocess_fullres(image, guidance)
        result = generator(net_input)
        result = processor.postprocess(result, pad_values)
        yield result


def save_results(input_imgs, results, output_path: Path):
    output_path.mkdir(exist_ok=True)
    for inp_path, res in zip(input_imgs, results):
        out_path = output_path / inp_path.name
        res = np.clip(res, 0, 1)
        res = (res * 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), res)


def main(args):
    
    downscale_model = tf.keras.models.load_model(args.downscale_model)

    if args.downscale_patch_size is not None:
        infer_provider = PatchInferenceProvider(downscale_model, (args.downscale_patch_size, args.downscale_patch_size), 
                                                (args.downscale_overlap_size, args.downscale_overlap_size))
        processor = InferenceProcessor(args.downscale_patch_size, args.downscale_factor)
    else:
        infer_provider = BasicInfrenceProvider(downscale_model)
        processor = InferenceProcessor(args.pad_factor, args.downscale_factor)
    # Getting sorted list of input images for each dir in input
    input_imgs = sum((list(sorted(Path(x).glob('*'))) for x in args.input), [])

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    results = list(tqdm(infer_downscale_model(infer_provider, input_imgs, processor), total = len(input_imgs)))

    save_results(input_imgs, results, output_path / "downscale")

    if args.fullres_model is None:
        return # Exit if fullres modelnot provided

    fullres_model = tf.keras.models.load_model(args.fullres_model)

    if args.fullres_patch_size is not None:
        infer_provider = PatchInferenceProvider(fullres_model, (args.fullres_patch_size, args.fullres_patch_size), 
                                                (args.fullres_overlap_size, args.fullres_overlap_size))
        processor = InferenceProcessor(args.fullres_patch_size, args.downscale_factor)
    else:
        infer_provider = BasicInfrenceProvider(fullres_model)
        processor = InferenceProcessor(args.pad_factor, args.downscale_factor)

    results = tqdm(infer_fullres_model(infer_provider, input_imgs, results, processor), total = len(input_imgs))

    save_results(input_imgs, results, output_path / "fullres")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dm', '--downscale_model', required=True, help='Downscale model path')
    parser.add_argument('-fm', '--fullres_model', required=False, default=None, help='Fullres model path')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Input path list')
    parser.add_argument('-o', '--output', required=True, help='Output path')

    parser.add_argument('-pf', '--pad_factor', required=False, default=4, type=int, help='Padding factor')
    parser.add_argument('-df', '--downscale_factor', required=False, default=8, type=int, 
                        help='Downscale factor fow downsace model')
    parser.add_argument('-dps', '--downscale_patch_size', required=False, default=None, type=int, 
                        help='Patch size for downscale model. If set - patch based inference is used')                    
    parser.add_argument('-dos', '--downscale_overlap_size', required=False, default=0, type=int, 
                        help='Overlap size for downscale model used for patch based inference')  
    parser.add_argument('-fps', '--fullres_patch_size', required=False, default=None, type=int, 
                        help='Patch size for fullres model. If set - patch based inference is used')                    
    parser.add_argument('-fos', '--fullres_overlap_size', required=False, default=0, type=int, 
                        help='Overlap size for fullres model used for patch based inference')    
    args = parser.parse_args()
    main(args)
