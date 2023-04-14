from abc import ABC, abstractmethod


class AbstractTrainFactory(ABC):

    @staticmethod
    @abstractmethod
    def make_loss():
        ...

    @staticmethod
    @abstractmethod
    def make_optimizer():
        ...

    @staticmethod
    @abstractmethod
    def make_model():
        ...

    @staticmethod
    @abstractmethod
    def make_checkpoint():
        ...

    @staticmethod
    @abstractmethod
    def make_metrics_writer():
        ...

    @staticmethod
    @abstractmethod
    def make_files_writer():
        ...

    @staticmethod
    @abstractmethod
    def make_augmentations():
        ...

    @staticmethod
    @abstractmethod
    def make_dataset(transforms):
        ...

    @staticmethod
    @abstractmethod
    def make_train_step():
        ...