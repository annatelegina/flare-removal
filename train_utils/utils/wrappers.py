import tensorflow as tf
from tqdm import tqdm
from train_utils.utils.dataset import DataGenerator


class DataGenWrapper(tf.keras.utils.Sequence):
    def __init__(self, data_gen: DataGenerator, path_processor):
        self._data_gen = data_gen
        self._path_processor = path_processor

    @classmethod
    def from_paths(cls, paths_dataset: tf.data.Dataset, path_processor, crop_size: int, batch_size: int):
        print('Loading data')
        data = tuple(zip(*[tuple(map(lambda x: x.numpy(), path_processor.load(*x))) for x in tqdm(paths_dataset)]))
        data_gen = DataGenerator(data, crop_size, batch_size, pad=0)
        return cls(data_gen, path_processor)

    def __len__(self) -> int:
        return len(self._data_gen)
    
    def on_epoch_end(self):
        self._data_gen.on_epoch_end()

    def __getitem__(self, index) -> list[tf.Tensor]:
        batch = self._data_gen[index]
        ret = []
        for args in zip(*batch):
            ret.append(self._path_processor.aug(args))
        ret = list(zip(*ret)) # batch, feature, .. -> feature, batch ...

        return tuple(tf.convert_to_tensor(x) for x in ret) # We covert features independently as they can have various dtype

    def __call__(self):
        """
            Used to create generator in get_tf_dataset function
        """
        for i in range(len(self)):
            yield self[i]

    def get_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(self,  output_signature=self._path_processor.OUT_SPEC)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(self)))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
