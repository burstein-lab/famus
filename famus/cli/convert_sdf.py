from famus.sdf import load
import pickle
import os
import sys
import argparse
from famus import config
from .common import get_common_parser
from .common_model_args import get_common_model_args_parser
import yaml


def main():
    prog = os.path.basename(sys.argv[0])
    if prog.endswith(".py"):
        prog = "python -m famus.cli.convert_sdf"
    parser = argparse.ArgumentParser(
        parents=[get_common_parser(), get_common_model_args_parser()],
        description="Convert sdf_train.json files of installed models to pickle format.",
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example usage:

  # {prog}

  Full description of this module can be found at https://github.com/burstein-lab/famus

        """,
    )
    args = parser.parse_args()
    config_path = args.config
    cfg = config.load_cfg(config_path) if config_path else config.get_default_config()
    models_dir = args.models_dir or cfg["models_dir"]

    for model_type in ["comprehensive", "light"]:
        model_type_root_path = os.path.join(models_dir, model_type)
        for model in os.listdir(model_type_root_path):
            print(f"Checking {model_type} {model}")
            model_path = os.path.join(model_type_root_path, model)
            data_dir_path = os.path.join(model_path, "data_dir")
            sdf_json_path = os.path.join(data_dir_path, "sdf_train.json")
            if not os.path.exists(sdf_json_path):
                print(
                    f"{sdf_json_path} does not exist - has the model finished preprocessing?"
                )
                continue
            sdf_pkl_path = os.path.join(data_dir_path, "sdf_train.pkl")
            if not os.path.exists(sdf_pkl_path):
                print(f"Converting {sdf_json_path} to {sdf_pkl_path}")
                sdf = load(sdf_json_path)
                with open(sdf_pkl_path, "wb") as f:
                    pickle.dump(sdf, f)
            else:
                print(f"{sdf_pkl_path} already exists")


if __name__ == "__main__":
    main()
