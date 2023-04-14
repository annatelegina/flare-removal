import os

import tensorflow as tf

import numpy as np
import cv2
from tqdm import tqdm


def get_batch_from_one_image(image):
    image = tf.cast(image, tf.float32)
    # get slices along the 0-th axis
    image = tf.reshape(image, [4, 612, 3264, -1])
    # h/patch_h, w, patch_h, c
    image = tf.transpose(image, [0, 2, 1, 3])
    # get slices along the 1-st axis
    # h/patch_h, w/patch_w, patch_w,patch_h, c
    image = tf.reshape(image, [4, 4, 816, 612, -1])
    # num_patches, patch_w, patch_h, c
    image = tf.reshape(image, [4 * 4, 816, 612, -1])
    # num_patches, patch_h, patch_w, c
    return tf.transpose(image, [0, 2, 1, 3])


def get_images_array(folders_list, image_reader, images_extension='.png'):
    """
    Creates np.array of images of given type.
    :param folders_list: list of paths to folders containing images to process
    :param image_reader: Image_reader class instance
    """
    images_arrays = []
    for folder_name in folders_list:
        images_filenames = [os.path.join(folder_name, filename) for filename in os.listdir(folder_name)]
        images_filenames.sort()
        for filename in tqdm(images_filenames):
            if filename.endswith(images_extension):
                image = image_reader.get_transformed_image(filename)
                images_arrays.append(image)

    return images_arrays



def get_burst_array(folders_list, image_reader, images_extension='.raw', burst_size=6):
    """
    Creates np.array of images of given type.
    :param folders_list: list of paths to folders containing images to process
    :param image_reader: Image_reader class instance
    """
    images_arrays = []
    for folder_name in folders_list:
        images_filenames = [os.path.join(folder_name, filename) for filename in os.listdir(folder_name)]
        images_filenames.sort()
        burst = []
        for num, filename in tqdm(list(enumerate(images_filenames))):
            if filename.endswith(images_extension):
                image = image_reader.get_transformed_image(filename)
                image = image[:,:,0]
                burst.append(image)
                if ((num+1) % burst_size)==0:
                    burst = np.stack(burst, axis=-1)
                    images_arrays.append(burst)
                    burst=[]                  

    images_arrays = np.stack(images_arrays)
    images_arrays.astype(np.float32)
    return images_arrays


def get_filenames_array(folders_list, images_extension='.png'):
    """
    Creates np.array of path of images.
    :param folders_list: list of paths to folders containing images to process
    :param images_extension: selected file extensions
    """
    filenames_arrays = []
    for folder_name in folders_list:
        images_filenames = [os.path.join(folder_name, filename) for filename in os.listdir(folder_name)]
        images_filenames.sort()
        for filename in tqdm(images_filenames):
            if filename.endswith(images_extension):
                filenames_arrays.append(filename)

    return filenames_arrays


def get_random_corner(image_size=(2448, 3264), patch_size=512, pad=25):
    #print(image_size)
    assert (patch_size % 2) == 0
    #assert (image_size[0] % 2) == 0
    #assert (image_size[1] % 2) == 0

    allowed = (image_size[0] - patch_size, image_size[1] - patch_size)
    coords = (np.random.randint(0 + pad, allowed[0] // 2 + 1 - pad) * 2,
              np.random.randint(0 + pad, allowed[1] // 2 + 1 - pad) * 2)
    return coords



def get_patch(image, top_left, patch_size=512):
    
    patch = image[top_left[0]:top_left[0] + patch_size, top_left[1]:top_left[1] + patch_size, :]
    
    return patch


class DataGenerator(tf.keras.utils.Sequence):
    """Generates batches of random crops with specified size"""

    def __init__(self, dataset, patch_size=256, batch_size=4, shuffle=True, pad=25):
        """

        :param dataset: tuple of images arrays
        :param patch_size: int, size of the random patch
        :param batch_size: int, size of generate batch
        #:param image_size: tuple of ints, height and width of the image
        :param shuffle: Boolean, shuffle data after epoch end
        :param pad: int, use margin from border of the image when cropping random patch
        """
        'Initialization'
        self.dataset = dataset
        self.patch_size = patch_size
        self.batch_size = batch_size
        #self.image_size = image_size
        self.list_IDs = list(range(len(dataset[0])))
        self.shuffle = shuffle
        self.pad = pad        
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        max_index = min((index + 1) * self.batch_size, len(self.list_IDs))
        indexes = self.indexes[index * self.batch_size:max_index]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        batches_list = [[] for _ in range(len(self.dataset))]
        # Generate data
        for ID in list_IDs_temp:            
            corner = get_random_corner(image_size=self.dataset[0][ID].shape[:2], patch_size=self.patch_size, pad=self.pad)
            patches = [get_patch(self.dataset[img_array_index][ID], corner, patch_size=self.patch_size)
                       for img_array_index in range(len(self.dataset))]
            for img_array_index in range(len(self.dataset)):
                batches_list[img_array_index].append(patches[img_array_index])

        return np.array(batches_list)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            

class OnlineDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of random crops with specified size"""

    def __init__(self, dataset, image_reader, patch_size=32, batch_size=32, image_size=(2448, 3264), shuffle=True, pad=25):
        """

        :param dataset: tuple of images filenames
        :param patch_size: int, size of the random patch
        :param batch_size: int, size of generate batch
        :param image_size: tuple of ints, height and width of the image
        :param shuffle: Boolean, shuffle data after epoch end
        :param pad: int, use margin from border of the image when cropping random patch
        """
        'Initialization'
        self.dataset = dataset
        self.image_reader = image_reader
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.list_IDs = list(range(len(dataset[0])))
        self.shuffle = shuffle
        self.pad = pad
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        max_index = min((index + 1) * self.batch_size, len(self.list_IDs))
        indexes = self.indexes[index * self.batch_size:max_index]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        batches_list = [[] for _ in range(len(self.dataset))]
        # Generate data
        for ID in list_IDs_temp:
            corner = get_random_corner(image_size=self.image_size, patch_size=self.patch_size, pad=self.pad)
            patches = [get_patch(self.image_reader.get_transformed_image(self.dataset[img_array_index][ID]), corner, patch_size=self.patch_size)
                       for img_array_index in range(len(self.dataset))]
            for img_array_index in range(len(self.dataset)):
                batches_list[img_array_index].append(patches[img_array_index])

        batches_list = np.array(batches_list)
        return batches_list

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)





class DataGenerator_with_single_blend(tf.keras.utils.Sequence):
    """Generates batches of random crops with specified size"""

    def __init__(self, dataset, patch_size=256, batch_size=4, image_size=(2448, 3264), shuffle=True, pad=25):
        """

        :param dataset: tuple of images arrays
        :param patch_size: int, size of the random patch
        :param batch_size: int, size of generate batch
        :param image_size: tuple of ints, height and width of the image
        :param shuffle: Boolean, shuffle data after epoch end
        :param pad: int, use margin from border of the image when cropping random patch
        """
        'Initialization'
        self.dataset = dataset
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.list_IDs = list(range(len(dataset[0])))
        self.shuffle = shuffle
        self.pad = pad
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        max_index = min((index + 1) * self.batch_size, len(self.list_IDs))
        indexes = self.indexes[index * self.batch_size:max_index]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        batches_list = [[] for _ in range(len(self.dataset))]
        # Generate data
        for ID in list_IDs_temp:
            corner = get_random_corner(image_size=self.image_size, patch_size=self.patch_size, pad=self.pad)
            patches = [get_patch(self.dataset[img_array_index][ID], corner, patch_size=self.patch_size)
                       for img_array_index in range(len(self.dataset))]
            
            switch = np.random.randint(2)
            if switch ==1:
                print(switch)
                patches[0] = np.concatenate((patches[0][:,:,0,np.newaxis],patches[0][:,:,0,np.newaxis],patches[0][:,:,0,np.newaxis],patches[0][:,:,0,np.newaxis],patches[0][:,:,0,np.newaxis],
                                           patches[0][:,:,0,np.newaxis]), axis =-1)
            
            
            print(patches[0].shape)
            
            
            for img_array_index in range(len(self.dataset)):
                batches_list[img_array_index].append(patches[img_array_index])

        #batches_list = np.array(batches_list)
        return batches_list

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)



