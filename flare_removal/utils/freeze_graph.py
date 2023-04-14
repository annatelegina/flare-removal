import argparse
import logging
from pathlib import Path

import tensorflow as tf


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

logging.basicConfig(level=logging.INFO)


def freeze_graph(model, name, path, out_name):
    if model:
        logging.info('\n'+str(model.output)+'\n')

        logging.info("output op", out_name)
        input_meta_graph = f"{path}/{name}.meta"
    else:
        out_name = "tf_op_layer_Tanh"
        input_meta_graph = None

    from tensorflow.python.tools import freeze_graph

    freeze_graph.freeze_graph(input_graph=None,
                              input_saver=None,
                              input_binary=True,
                              input_checkpoint=f"{path}/{name}",
                              output_node_names=out_name,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=f"{path}/{name}.pb",
                              clear_devices=True,
                              initializer_nodes="",
                              input_meta_graph=input_meta_graph)


def main(args):
    checkpoints_path = Path(args.checkpoints_path)

    logging.info(f"save checkpoint to {checkpoints_path.as_posix()}")

    if not checkpoints_path.exists():
        logging.info(f"mkdir checkpoints folder {checkpoints_path.as_posix()}")

    if not checkpoints_path.exists():
        checkpoints_path.mkdir()

    tf.keras.backend.set_learning_phase(0)

    trained_model =  tf.keras.models.load_model(args.model)
    trained_model.summary()
    model_name = args.model.split('.')[0]
    checkpoint = tf.compat.v1.train.Saver()

    path = checkpoints_path.as_posix()

    checkpoint.save(
        tf.compat.v1.keras.backend.get_session(), (checkpoints_path / model_name).as_posix()
    )
    # logger.info('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node]))

    freeze_graph(trained_model, model_name, path, args.out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--out_name', required=True)
    parser.add_argument('-ckpt', '--checkpoints_path', required=False, default="./tmp_checkpoints")

    args = parser.parse_args()
    with tf.device("/cpu:0"):
        main(args)

