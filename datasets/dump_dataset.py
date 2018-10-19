import argparse
import tensorflow as tf
import keras as K
import numpy as np
import tfrecord_dataset

tf.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser(
        description="Convert json-based input file into tfrecord-based.")
    parser.add_argument("source", metavar = "SOURCE",
        help = "Path to .tfrecords file")
    parser.add_argument("-c", "--count", type = int, help = "Number of records to dump")
    parser.add_argument("-n", "--number", type = int, help = "Dump record at specific number")
    parser.add_argument("-l", "--label", type = int, help = "Filter records with only this label")

    args = parser.parse_args()

    tf_record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)

    count = 0
    for record in tf.python_io.tf_record_iterator(
            args.source, 
            tf_record_options):

        example = tf.parse_single_example(
            record,
            features = tfrecord_dataset.features
        )
        if args.count and count >= args.count:
                return
        label = tf.cast(example.pop("label"), tf.int32)
        compare_result = np.array(tf.not_equal(label, tf.constant(args.label)))
        if args.label is not None and compare_result:
                continue
        if args.number and count != args.number:
                count = count + 1
                continue
        domain = tfrecord_dataset.tensor_to_string(tf.decode_raw(example["domain"], tf.int32))
        url = tfrecord_dataset.tensor_to_string(tf.decode_raw(example["url"], tf.int32))
        print(count, ": ", url, domain, label, "\n")
        if args.number and count == args.number:
            return

        count = count + 1


if __name__ == "__main__":
    main()