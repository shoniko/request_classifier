import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
from datasets import tfrecord_dataset

def run_prediction(dataset, url_classifier, config):
    label = 0
    model_dir = os.path.join(os.getcwd(), "output", config["network"])
    train_input_fn = lambda: dataset.predict_input_fn(config["input"]["predict_dataset_path"],
        config["batch_size"], config["input"]["compression"],
        equals = tf.constant(label)
        )
    predictions = url_classifier.predict(input_fn = train_input_fn)
    num = 0
    for pred_dict in predictions:
        if pred_dict["class_ids"][0] != label:
            print(num)
            print(pred_dict)
        num = num + 1


def start_training(dataset, url_classifier, config, profile = False):

    model_dir = os.path.join(os.getcwd(), "output", config["network"])
    beholder = Beholder(model_dir)
    beholder_hook = BeholderHook(model_dir)
    hooks = [
        # These are for debugging mostly. Uncomment when needed
#        tf_debug.LocalCLIDebugHook(),
#        beholder_hook,
#         tf_debug.TensorBoardDebugHook("localhost:6009")
        ]
    train_input_fn = lambda: dataset.train_input_fn(config["input"]["train_dataset_path"],
        config["batch_size"], config["input"]["compression"]
        )
    eval_input_fn = lambda: dataset.train_input_fn(config["input"]["eval_dataset_path"],
        config["batch_size"], config["input"]["compression"]
        )
    if profile:
        # This is only for profiling / debugging
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()
        with tf.contrib.tfprof.ProfileContext(config["profile_path"],
            trace_steps=range(998, 1003), dump_steps=[10, 999, 1000, 1001, 1500]) as pctx:
            print("Training")
            url_classifier.train(input_fn = train_input_fn, steps = 2000)
            print("Evaluating")
            eval_result = url_classifier.evaluate(input_fn = eval_input_fn)
            print(eval_result)

    for epoch in range(1, config["epochs"]):
        print("Training")
        url_classifier.train(input_fn = train_input_fn, steps = config["train_steps"])
        print("Evaluating")
        eval_result = url_classifier.evaluate(input_fn = eval_input_fn, steps = config["eval_steps"])
        print(eval_result)
