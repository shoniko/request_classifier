import argparse
import subprocess
from experiment import create_experiment
import tensorflow as tf

tf.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser(description='Make predictions.')
    parser.add_argument("-c", "--config", help='Path to a config file', required=True)

    args = parser.parse_args()
    
    create_experiment(args.config, True)
    
if __name__ == '__main__':
    main()