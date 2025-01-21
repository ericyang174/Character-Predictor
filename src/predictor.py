import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('-c', '--config', required=True, help="Path to configuation")
    parser.add_argument('-m', '--mode', required=True, help="Mode should be either train or test")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    mode = args.mode 

    print(f"Configuration: {config}")
    print(f"Mode: {mode}")

if __name__ == '__main__':
    main()