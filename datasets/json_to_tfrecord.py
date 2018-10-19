import argparse
import json
import tensorflow as tf
import keras as K
import keras.preprocessing
import numpy as np
import os

TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1

def _int64_feature(value):
    lst = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=lst)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    parser = argparse.ArgumentParser(
        description="Convert json-based input file into tfrecord-based.")
    parser.add_argument("source", metavar = "SOURCE",
        help = "Path to source file (list of JSON records)")
    parser.add_argument("destination", metavar = "DESTINATION",
        help = "Path to output directory.")
    parser.add_argument("-s", "--split", action="store_true",
        help = "[Optional] Split the dataset 80/10/10")

    args = parser.parse_args()

    counter = 0
    positive = 0
    negative = 0
    elemhide = 0

    tokenizer = K.preprocessing.text.Tokenizer(char_level = True, filters = None, oov_token = -1)
    tokenizer.fit_on_texts("abcdefghijklmnopqrstuvwxyz1234567890:/?=+.")

    tf_record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)

    destination_path = os.path.join(args.destination, "result.tfrecords")
    with open(args.source) as infile:
        with tf.python_io.TFRecordWriter(destination_path, tf_record_options) as writer:
            for line in infile:
                row = json.loads(line)
                url = ""
                if "url" in row["request"]:
                    url = row["request"]["url"]
                else:
                    # Skip ELEMHIDE records
                    elemhide += 1
                    continue
                if row["filter"] is None:
                    label = 0
                    negative += 1
                else:
                    label = 1
                    positive += 1
                url = K.preprocessing.sequence.pad_sequences(
                    [tokenizer.texts_to_sequences(url)],
                    maxlen = 2048, padding = "post"
                    )
                url = np.concatenate(np.array(url)).ravel()
                reqType = row["request"]["type"]
                docDomain = row["request"]["docDomain"]
                docDomain = K.preprocessing.sequence.pad_sequences(
                    [tokenizer.texts_to_sequences(docDomain)],
                    maxlen = 100, padding = "post"
                    )
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            "url": _bytes_feature(np.matrix(url).A1.tostring()),
                            "type": _bytes_feature(tf.compat.as_bytes(reqType)),
                            "domain": _bytes_feature(np.matrix(docDomain).A1.tostring()),
                            "label": _int64_feature(label)
                        }
                    )
                )
                writer.write(example.SerializeToString())
                counter += 1
                if counter % 10000 == 0:
                    print("Records processed: ", counter)
            writer.close()

    print("\n---\n")
    print("Written records: ", counter)
    print("Positive examples: ", positive)
    print("Negative examples: ", negative)
    print("ELEMHIDE hits: ", elemhide)
    print("---\n")

    if args.split:
        train_size = int(counter * TRAIN_RATIO)
        eval_size = int(counter * EVAL_RATIO)
        test_size = int(counter * TEST_RATIO)
        print("Splitting to dataset: ", train_size, eval_size, test_size)
        eval_steps = EVAL_RATIO * 100
        test_steps = TEST_RATIO * 100
        steps_since_eval = eval_steps - 5
        steps_since_test = 0

        train_records = 0
        test_records = 0
        eval_records = 0
        train_destination = os.path.join(args.destination, "result_train.tfrecords")
        eval_destination = os.path.join(args.destination, "result_eval.tfrecords")
        test_destination = os.path.join(args.destination, "result_test.tfrecords")

        with tf.python_io.TFRecordWriter(
                train_destination, tf_record_options
            ) as train_writer, tf.python_io.TFRecordWriter(
                eval_destination, tf_record_options
            ) as eval_writer, tf.python_io.TFRecordWriter(
                test_destination, tf_record_options
            ) as test_writer:

            for record in tf.python_io.tf_record_iterator(
                    destination_path, 
                    tf_record_options):
                steps_since_eval = steps_since_eval + 1
                steps_since_test = steps_since_test + 1
                if steps_since_eval == eval_steps:
                    # Write to eval set
                    eval_writer.write(record)
                    steps_since_eval = 0
                    eval_records = eval_records + 1
                    continue
                if steps_since_test == test_steps:
                    # Write to test set
                    test_writer.write(record)
                    steps_since_test = 0
                    test_records = test_records + 1
                    continue
                # Write to train set
                train_writer.write(record)
                train_records = train_records +1
        print("Split dataset. Train/eval/test", train_records, eval_records, test_records)

if __name__ == "__main__":
    main()