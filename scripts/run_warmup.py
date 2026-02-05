import argparse
import yaml
from utils.seed import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # set seed
    set_seed(cfg["seed"])

    print("Config loaded successfully")
    print(cfg)


if __name__ == "__main__":
    main()