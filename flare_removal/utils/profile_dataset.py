from train_utils.utils import dataset_tf, path_processors
import albumentations as A
from flare_removal.data_config import TRAIN_FOLDERS
import cv2
import numpy as np
from train_utils.train.train_setup import setup_gpu
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train_utils.utils.wrappers import DataGenWrapper
IMAGE_SHAPE = np.array([3648, 2736])

GPU = 0


def main():
    setup_gpu(GPU)
    transforms = A.Compose(
        [A.ShiftScaleRotate(scale_limit=[-0.65, 0.25], rotate_limit=90, interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT_101, p=0.8),
         A.HueSaturationValue(hue_shift_limit=20 / 255, sat_shift_limit=20 / 255, val_shift_limit=30 / 255,
                              always_apply=False, p=0.25),
         A.ChannelShuffle(p=0.25)],
        additional_targets={'image0': 'image', 'image1': 'image'})
    # path_processor = path_processors.FullresPathProcessor(transforms=transforms,
    #                                                  min_alpha=1,
    #                                                  r=100)
    path_processor = path_processors.BasePathProcessor(resize_shape=IMAGE_SHAPE // 8,
                                                    transforms=transforms,
                                                     min_alpha=1,
                                                     r=100)

    old_gen = dataset_tf.get_dataset(data_paths=TRAIN_FOLDERS)

    # train_generator = train_generator.repeat(10)
    
    # train_generator = old_gen.map(path_processor.load, num_parallel_calls=tf.data.AUTOTUNE)
    # X, Y = [x.numpy() for x, _ in train_generator], [x.numpy() for _, x in train_generator]
    # dataset = DataGenerator((X, Y), 324, 4, pad=0)
    dataset_ori = DataGenWrapper.from_paths(old_gen, path_processor, 224, 4)
    dataset = dataset_ori.get_tf_dataset()
    # print(sys.getsizeof(dataset[0][0]))
    # print(sys.getsizeof(dataset[0]))
    # print(sys.getsizeof(dataset[0]))
    # train_generator = train_generator.cache()

    # manager = TFRecordManager('test.tfrecord', ['flare', 'gt'])
    # train_generator = manager.load()
    

    # print(len(train_generator))
    # train_generator = train_generator.shuffle(buffer_size=165)
    
    # train_generator = train_generator.prefetch(tf.data.AUTOTUNE)

    # train_generator = train_generator.map(path_processor.aug, num_parallel_calls=tf.data.AUTOTUNE)
    # train_generator = train_generator.batch(batch_size=4)
    # train_generator = train_generator.prefetch(tf.data.AUTOTUNE)
    # print(len(train_generator))

    # tf.profiler.experimental.start('logdir')
    # print(len(dataset))
    for e in range(2):
        
        for i, b in enumerate(tqdm(dataset)):   
            # print(b)
            # print("|".join([str(x.shape) for x in  b]))

            # print(b[0].shape, b[1].shape)
            for item in zip(*b):
                plt.imsave(f'tmp/profile/{e}_{i}_0.jpg', item[0][..., :3].numpy())
                plt.imsave(f'tmp/profile/{e}_{i}_1.jpg', item[1].numpy())
            pass
            dataset_ori.on_epoch_end()
    # tf.profiler.experimental.stop()


if __name__ == '__main__':
    main()
