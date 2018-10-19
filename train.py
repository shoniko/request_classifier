import argparse
import subprocess
from experiment import create_experiment

def main():
    parser = argparse.ArgumentParser(description = 'Start training.')
    parser.add_argument("-c", "--config", help = 'Path to a config file', required = True)

    args = parser.parse_args()
    
    create_experiment(args.config)
    
if __name__ == '__main__':
    main()