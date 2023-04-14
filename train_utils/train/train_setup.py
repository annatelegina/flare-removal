import git
import tensorflow as tf


def check_repo():
    repo = git.Repo(search_parent_directories=True)

    if repo.is_dirty():
        answer = input('\033[93m'+"You have uncommitted changes, do you want to proceed? \nY/N \n"+'\033[0m')
        while answer != "Y" or answer != "N":
            if answer == "Y":
                break
            elif answer == "N":
                raise KeyboardInterrupt
            else:
                answer = input("Please choose Y or N \n")

    sha = repo.head.object.hexsha
    return sha


def setup_gpu(gpu_index):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_index], True)
    tf.config.experimental.set_visible_devices(physical_devices[gpu_index], 'GPU')
    print("GPU {} will be used for training".format(gpu_index))
