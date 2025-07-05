from src.pipeline import Chatbot_Pipeline
import yaml
from pathlib import Path
import argparse

def main(config):
    pipeline = Chatbot_Pipeline(config)
    if config['training']:
        pipeline.train()
    else:
        pipeline.test()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="yamls/train.yaml",
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)
    