import tensorflow as tf
import keras as K
import numpy as np

features = {
            'url': tf.FixedLenFeature([], tf.string),
            'type': tf.FixedLenFeature([], tf.string),
            'domain': tf.FixedLenFeature([], tf.string),
            'label':tf.FixedLenFeature([], tf.int64)
            }

tokenizer = K.preprocessing.text.Tokenizer(char_level = True, filters = None, oov_token = -1)
tokenizer.fit_on_texts("abcdefghijklmnopqrstuvwxyz1234567890:/?=+.")
reverse_tokenizer_map = dict(map(reversed, tokenizer.word_index.items()))

def tensor_to_string(tensor):
    tensorInNumbers = np.array(tensor)
    nonZeroMask = np.greater(tensorInNumbers, 0)
    tensorInNumbers = np.extract(nonZeroMask, tensorInNumbers)
    return "".join([str(reverse_tokenizer_map[val]) for val in tensorInNumbers])

def get_example(source, number, label):
    tf_record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)
    count = 0
    for record in tf.python_io.tf_record_iterator(
            source, 
            tf_record_options):

        example = tf.parse_single_example(
            record,
            features = features
        )
        if tf.not_equal(example["label"], label):
            continue

        if count == number:
            domain = tensor_to_string(tf.decode_raw(example["domain"], tf.int32))
            url = tensor_to_string(tf.decode_raw(example["url"], tf.int32))

            return (url, domain, example["type"], example["label"])

        count = count + 1

def decode(serialized_example):
    example = tf.parse_single_example(
        serialized_example,
        features = features
    )
    label = tf.cast(example.pop("label"), tf.int32)
    domain = tf.decode_raw(example["domain"], tf.int32)
    url = tf.decode_raw(example["url"], tf.int32)
    return (dict({"url": url, "domain": domain, "type": example["type"]}), label)

def train_input_fn(file_path, batch_size = 1, compression = "", repeat = True, equals = None):
    dataset = tf.data.TFRecordDataset(file_path, compression)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(decode, num_parallel_calls = 8)
    if equals is not None:
        dataset = dataset.filter(lambda features, label:
            tf.equal(equals, label)
            )
    return dataset.batch(batch_size).prefetch(batch_size)

def predict_input_fn(file_path, batch_size = 1, compression = "", equals = None):
    dataset = tf.data.TFRecordDataset(file_path, compression)
    dataset = dataset.map(decode)
    if equals is not None:
        dataset = dataset.filter(lambda features, label:
            tf.equal(equals, label)
            )
    return dataset.batch(batch_size)