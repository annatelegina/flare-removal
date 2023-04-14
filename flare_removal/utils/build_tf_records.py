from typing import Tuple
import numpy as np
import tensorflow as tf
from fire import Fire
from flare_removal.data_config import TRAIN_FOLDERS
from tqdm import tqdm
from train_utils.utils import dataset_tf
from train_utils.utils.path_processors import (BasePathProcessor,
                                          FullresPathProcessor)

IMAGE_SHAPE = np.array([3648, 2736])


class TFRecordManager:
    def __init__(self, filename: str, fields: Tuple[str] = ("flare", "gt")):
        self._filename = filename
        self._fields = fields
    
    def write(self, generator):
        with tf.io.TFRecordWriter(self._filename) as file_writer:
            for args in tqdm(generator):
                features = {name: tf.train.Feature(bytes_list=tf.train.BytesList(value=[arg.numpy().tobytes()])) 
                                for name, arg in zip (self._fields, args)}
                record_bytes = tf.train.Example(features=tf.train.Features(feature={
                    **features,
                    "h": tf.train.Feature(int64_list=tf.train.Int64List(value=[args[0].shape[0]])),
                    "w": tf.train.Feature(int64_list=tf.train.Int64List(value=[args[0].shape[1]])),
                    "c": tf.train.Feature(int64_list=tf.train.Int64List(value=[args[0].shape[2]])),
                })).SerializeToString()
                file_writer.write(record_bytes)
    
    def load(self):
        feats = {name: tf.io.FixedLenFeature([], tf.string) for name in self._fields}
        feature_set = { **feats,
               'h': tf.io.FixedLenFeature([], tf.int64),
               'w': tf.io.FixedLenFeature([], tf.int64),
               'c': tf.io.FixedLenFeature([], tf.int64),
           }

        raw_dataset = tf.data.TFRecordDataset([self._filename])
        
        raw_dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, feature_set), num_parallel_calls=tf.data.AUTOTUNE)
        raw_dataset = raw_dataset.map(lambda x: [self._decode(x[name], x['h'], x['w'], x['c']) for name in self._fields], num_parallel_calls=tf.data.AUTOTUNE)
        return raw_dataset

    def _decode(self, raw, h, w, c):
        raw = tf.io.decode_raw(raw, tf.uint8)
        return tf.reshape(raw, [h, w, c])
    

def main(mode, filename):
    transforms = None
    if mode == 'downscale':
        path_processor = BasePathProcessor(resize_shape=IMAGE_SHAPE // 8,
                                                        transforms=transforms,
                                                        min_alpha=1,
                                                        r=100)
        manager = TFRecordManager(filename, ['flare', 'gt'])
    elif mode == 'fullres':
        path_processor = FullresPathProcessor(
                                                     transforms=transforms,
                                                     min_alpha=1,
                                                     r=100,)
        manager = TFRecordManager(filename, ['flare', 'gt'])
    else:
        raise Exception(f"Unknown mode: {mode}")

    train_generator = dataset_tf.get_dataset(data_paths=TRAIN_FOLDERS)
    
    train_generator = train_generator.map(path_processor.load, num_parallel_calls=tf.data.AUTOTUNE)
    manager.write(train_generator)




if __name__ == '__main__':
   Fire(main)
