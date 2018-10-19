import tensorflow as tf
import keras as K
import keras.preprocessing
from keras.preprocessing.text import Tokenizer
import json
import numpy as np


tokenizer = K.preprocessing.text.Tokenizer(char_level = True, filters = None, oov_token = -1)
tokenizer.fit_on_texts("abcdefghijklmnopqrstuvwxyz1234567890:/?=+.")

def processLine_tf(line):
    url, reqType, docDomain, label = tf.py_func(
            processLine_py, [line], 
            [tf.int32, tf.string, tf.int32, tf.int32]
        )
    return ({"url": url, "type": reqType, "domain": docDomain}, label)

def processLine_py(line):
    row = json.loads(line)
    label = [0]
    if row["filter"] is not None:
        label = [1]
    url = ""
    if "url" in row["request"]:
        url = row["request"]["url"]
    url = K.preprocessing.sequence.pad_sequences(
                    [tokenizer.texts_to_sequences(url)],
                    maxlen = 2048, padding = "post"
                    )
    url = np.concatenate(np.array(url)).ravel().astype(int)
    reqType = row["request"]["type"]
    docDomain = row["request"]["docDomain"]
    docDomain = K.preprocessing.sequence.pad_sequences(
        [tokenizer.texts_to_sequences(docDomain)],
        maxlen = 100, padding = 'post', truncating = "post"
        )
    docDomain = np.concatenate(np.array(docDomain)).ravel().astype(int)
    return url, reqType, docDomain, label

def train_input_fn(file_path, dataset_size, batch_size=1):
    datasource = tf.data.TextLineDataset(file_path)
    datasource = datasource.map(processLine_tf)
    ret = datasource.shuffle(1000).batch(batch_size)
    return ret

def eval_input_fn(file_path, dataset_size, batch_size=1):
    datasource = tf.data.TextLineDataset(file_path)
    datasource = datasource.map(processLine_tf)
    ret = datasource.batch(batch_size)
    return ret

def get_first_sample(file_path):
    datasource = tf.data.TextLineDataset(file_path)
    datasource = datasource.map(processLine_tf)
    it = datasource.make_one_shot_iterator()
    return it.get_next()
