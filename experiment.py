import os
import json
from datasets import json_dataset
from datasets import tfrecord_dataset
from datasets import json_dataset
from training import start_training
from training import run_prediction
import tensorflow as tf

alphabet_size = 0

def parse_config_file(config_path):
    with open(config_path) as f:
        config = json.load(f)
    global alphabet_size
    alphabet_size = len(config["input"]["alphabet"])
    return config

def char_norm(char):
    return char / alphabet_size

def build_feature_columns(config):
    urls = tf.feature_column.numeric_column(
        key = "url",
        shape = [2048],
        normalizer_fn = char_norm
    )

    request_type = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        key = "type", vocabulary_list = ["SCRIPT", "SUBDOCUMENT", "IMAGE", "XMLHTTPREQUEST",
        "FONT", "DOCUMENT", "STYLESHEET"]
    ))

    domain = tf.feature_column.numeric_column(
        key = "domain",
        shape = [100],
        normalizer_fn = char_norm,
    )

    features = [urls, request_type, domain]
    return features

def create_experiment(config_path, predict = False):
    config = parse_config_file(config_path)
    if config["input"]["type"] == "tfrecords":
        dataset = tfrecord_dataset
    elif config["input"]["type"] == "json":
        dataset = json_dataset
    
    warm_start_location = None
    if "warm_start_location" in config:
        warm_start_location = config["warm_start_location"]

    model_dir = os.path.join(os.getcwd(), "output", config["network"])
    os.makedirs(model_dir, exist_ok=True)

    if config["network"] == "cnn":
        network = CNN(config["network_config"])
        url_classifier = tf.keras.estimator.model_to_estimator(keras_model=network.model,
                                                    model_dir=model_dir)
    elif config["network"] == "lstm":
        raise "Not implemented"
    elif config["network"] == "fc":
        raise "Not implemented"
    elif config["network"] == "dnn_classifier_estimator":
        binary_head = tf.contrib.estimator.binary_classification_head()
        url_classifier = tf.contrib.estimator.DNNEstimator(
            feature_columns = build_feature_columns(config),
            hidden_units = config["network_config"]["fc_layers"],
            dropout = config["network_config"]["dropout"],
            head = binary_head,
            model_dir = model_dir,
            warm_start_from = warm_start_location,
            optimizer=lambda: tf.train.AdamOptimizer(
                learning_rate = tf.train.exponential_decay(
                            learning_rate = config["learning_rate"],
                            global_step = tf.train.get_global_step(),
                            decay_steps = 10000,
                            decay_rate = 0.96)
            )
        )
    elif config["network"] == "baseline_classifier_estimator":
        binary_head = tf.contrib.estimator.binary_classification_head()
        url_classifier = tf.contrib.estimator.BaselineEstimator(
            head = binary_head,
            model_dir=model_dir,
            optimizer=lambda: tf.train.AdamOptimizer(
                learning_rate = config["learning_rate"]
            )
        )
    elif config["network"] == "rnn_classifier_estimator":
        raise "Not implemented"
        # This needs more work
        url_classifier = tf.contrib.estimator.RNNClassifier(
            model_dir=model_dir,
            learning_rate = config["learning_rate"],
            feature_columns = build_feature_columns(config),
            n_trees = 100,
            max_depth = 6,
            n_batches_per_layer = 2
        )
    else:
        raise "Unsupported network in config"

    if predict:
        return run_prediction(dataset, url_classifier, config)
    return start_training(dataset, url_classifier, config)