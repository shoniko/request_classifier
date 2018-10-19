def main():
    parser = argparse.ArgumentParser(
        description="Split the tfrecords dataset into train, eval and test sets.")
    parser.add_argument("source", metavar = "SOURCE",
        help = "Source dataset")
    parser.add_argument("dest_train", metavar = "DESTTEST",
        help = "Destination for train dataet")
    parser.add_argument("dest_eval", metavar = "DESTTEST",
        help = "Destination for eval dataet")
    parser.add_argument("dest_test", metavar = "DESTTEST",
        help = "Destination for test dataet")
    parser.add_argument("train", metavar = "TRAIN",
        help = "Percentage of records to go into a train dataset")
    parser.add_argument("eval", metavar = "EVAL",
        help = "Percentage of records to go into a train dataset")
    parser.add_argument("test", metavar = "TEST",
        help = "Percentage of records to go into a test dataset")
    parser.add_argument("size", metavar = "SIZE",
        help = "Optional size of the dataset")

    args = parser.parse_args()
    if args.train + args.eval + args.test != 100:
        print("Sum of all perecntages is expected to be 100")
    
    dataset_size = args.size
    if dataset_size is None:
        dataset_size = 0
        for record in tf.python_io.tf_record_iterator(path = file_path):
            dataset_size = dataset_size + 1

    train_size = dataset_size / 100 * args.train
    eval_size = dataset_size / 100 * args.eval
    test_size = dataset_size / 100 * args.test

    for i in range(1, dataset_size)
if __name__ == "__main__":
    main()