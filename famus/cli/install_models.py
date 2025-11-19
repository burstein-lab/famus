import argparse
import os
import shutil
import sys
import tarfile
import gzip
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

from famus.logging import setup_logger, logger
from .common import get_common_parser
from famus import config

ZENODO_BASE = "https://zenodo.org/records/14941374/files"

AVAILABLE_MODELS = ["kegg", "orthodb", "interpro", "eggnog"]
AVAILABLE_TYPES = ["comprehensive", "light"]

TYPE_MAP = {"comprehensive": "full", "light": "light"}

MODEL_TO_THRESHOLD = {
    "kegg_comprehensive": 0.19,
    "kegg_light": 0.19,
    "orthodb_comprehensive": 0.19,
    "orthodb_light": 0.19,
    "interpro_comprehensive": 0.34,
    "interpro_light": 0.35,
    "eggnog_comprehensive": 0.19,
    "eggnog_light": 0.19,
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=os.path.basename(output_path)
    ) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def parse_model_spec(model_spec):
    """
    Parse model specification like 'kegg_comprehensive' or 'orthodb_light'
    Returns (model_name, model_type)
    """
    parts = model_spec.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model specification: {model_spec}. "
            f"Expected format: <model>_<type> (e.g., kegg_comprehensive, orthodb_light)"
        )

    model_name, model_type = parts

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {', '.join(AVAILABLE_MODELS)}"
        )

    if model_type not in AVAILABLE_TYPES:
        raise ValueError(
            f"Unknown type: {model_type}. Available: {', '.join(AVAILABLE_TYPES)}"
        )

    return model_name, model_type


def download_model_tar(model_name, model_type, download_dir):
    """
    Download model tar file from Zenodo
    Returns path to downloaded tar file
    """
    tar_type = TYPE_MAP[model_type]
    tar_filename = f"{tar_type}_{model_name}.tar"
    tar_url = f"{ZENODO_BASE}/{tar_filename}?download=1"
    tar_path = os.path.join(download_dir, tar_filename)

    if os.path.exists(tar_path):
        logger.info(f"Tar file already exists: {tar_path}")
        return tar_path

    logger.info(f"Downloading {tar_filename}...")
    try:
        download_url(tar_url, tar_path)
        logger.info(f"✓ Downloaded {tar_filename}")
        return tar_path
    except Exception as e:
        logger.error(f"Failed to download {tar_filename}: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        raise


def extract_model(tar_path, model_name, model_type, models_dir):
    """
    Extract and organize model files
    """
    # Create target directory structure
    target_dir = os.path.join(models_dir, model_type, model_name)
    data_dir = os.path.join(target_dir, "data_dir")
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Extracting {os.path.basename(tar_path)}...")

    # Extract tar file to temporary location
    temp_extract = os.path.join(models_dir, model_type, f".tmp_{model_name}")
    os.makedirs(temp_extract, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_extract)

        # Move extracted files to target location
        extracted_model_dir = os.path.join(temp_extract, model_name)
        if os.path.exists(extracted_model_dir):
            # Move all files from extracted directory to target
            for item in os.listdir(extracted_model_dir):
                src = os.path.join(extracted_model_dir, item)
                dst = os.path.join(target_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)

        # Cleanup temp directory
        shutil.rmtree(temp_extract)

        # Decompress files
        logger.info(f"Decompressing files for {model_name}_{model_type}...")

        # SDF file (naming differs by type)
        if model_type == "light":
            sdf_gz = os.path.join(target_dir, f"{model_name}_sdf_light.json.gz")
        else:
            sdf_gz = os.path.join(target_dir, f"{model_name}_sdf.json.gz")

        if os.path.exists(sdf_gz):
            sdf_json = os.path.join(data_dir, "sdf_train.json")
            logger.info(f"  → sdf_train.json")
            with gzip.open(sdf_gz, "rb") as f_in:
                with open(sdf_json, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(sdf_gz)

        # Train embeddings
        embeddings_gz = os.path.join(target_dir, "train_embeddings.npy.gz")
        if os.path.exists(embeddings_gz):
            embeddings_npy = os.path.join(data_dir, "train_embeddings.npy")
            logger.info(f"  → train_embeddings.npy")
            with gzip.open(embeddings_gz, "rb") as f_in:
                with open(embeddings_npy, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(embeddings_gz)

        # Model state
        state_gz = os.path.join(target_dir, "state.pt.gz")
        if os.path.exists(state_gz):
            state_pt = os.path.join(target_dir, "state.pt")
            logger.info(f"  → state.pt")
            with gzip.open(state_gz, "rb") as f_in:
                with open(state_pt, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(state_gz)

        # Subcluster profiles (both comprehensive and light have these)
        # Light models have one profile per family, comprehensive have multiple
        profiles_tar_gz = os.path.join(target_dir, "subcluster_profiles.tar.gz")
        if os.path.exists(profiles_tar_gz):
            logger.info(f"  → subcluster_profiles/")
            shutil.move(profiles_tar_gz, data_dir)
            with tarfile.open(
                os.path.join(data_dir, "subcluster_profiles.tar.gz"), "r:gz"
            ) as tar:
                tar.extractall(path=data_dir)
            os.remove(os.path.join(data_dir, "subcluster_profiles.tar.gz"))

        with open(os.path.join(target_dir, "env"), "w") as f:
            f.write(
                "THRESHOLD=" + str(MODEL_TO_THRESHOLD[f"{model_name}_{model_type}"])
            )
        logger.info(f"Installed {model_name}_{model_type} successfully.")

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(temp_extract):
            shutil.rmtree(temp_extract)
        raise e


def main():
    prog = os.path.basename(sys.argv[0])
    if prog.endswith(".py"):
        prog = "python -m famus.cli.install_models"
    parser = argparse.ArgumentParser(
        parents=[get_common_parser()],
        description="Download and install FAMUS pre-trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog=prog,
        epilog=f"""
Examples:

  # Install specific models
  {prog} --models kegg_comprehensive orthodb_light
  
  # Install all comprehensive models
  {prog} --models kegg_comprehensive orthodb_comprehensive interpro_comprehensive eggnog_comprehensive
  
  # Install to custom directory
  {prog} --models kegg_light --models-dir /path/to/my/models
  
  # Keep downloaded tar files
  {prog} --models kegg_light --keep-tars

Available models: {", ".join(AVAILABLE_MODELS)}
Available types: {", ".join(AVAILABLE_TYPES)}

Format: <model>_<type> (e.g., kegg_comprehensive, orthodb_light)

Full description of arguments can be found at https://github.com/burstein-lab/famus
        """,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory to save the installed models to",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to install (format: model_type, e.g., kegg_comprehensive orthodb_light)",
    )

    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Keep downloaded tar files after extraction",
    )

    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory to download tar files to (default: current directory)",
    )

    args = parser.parse_args()
    cfg_path = args.config
    cfg = config.load_cfg(cfg_path) if cfg_path else config.get_default_config()
    no_log = args.no_log or cfg["no_log"]
    log_dir = args.log_dir or cfg["log_dir"]
    models_dir = args.models_dir or cfg["models_dir"]
    logger = setup_logger(enable_logging=not no_log, log_dir=log_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "comprehensive").mkdir(exist_ok=True)
    (models_dir / "light").mkdir(exist_ok=True)
    download_dir = args.download_dir if args.download_dir else Path.cwd()
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Download directory: {download_dir}")
    models_to_install = []
    try:
        for model_spec in args.models:
            model_name, model_type = parse_model_spec(model_spec)
            models_to_install.append((model_name, model_type))
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Installing {len(models_to_install)} model(s)\n")
    successful = []
    failed = []
    for model_name, model_type in models_to_install:
        model_spec = f"{model_name}_{model_type}"
        target_dir = models_dir / model_type / model_name
        if (target_dir / "env").exists():
            logger.info(f"{model_spec} already installed")
            successful.append(model_spec)
            continue
        try:
            logger.info(f"{'=' * 60}")
            logger.info(f"Installing {model_spec}")
            logger.info(f"{'=' * 60}")
            tar_path = download_model_tar(model_name, model_type, download_dir)
            extract_model(tar_path, model_name, model_type, models_dir)
            if not args.keep_tars:
                logger.info(f"Removing {os.path.basename(tar_path)}")
                os.remove(tar_path)
            successful.append(model_spec)
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to install {model_spec}: {e}")
            failed.append(model_spec)
            logger.info("")

    logger.info(f"{'=' * 60}")
    logger.info("INSTALLATION SUMMARY")
    logger.info(f"{'=' * 60}")

    if successful:
        logger.info(f"Successfully installed ({len(successful)}):")
        for model in successful:
            logger.info(f"  • {model}")

    if failed:
        logger.info(f"\nFailed ({len(failed)}):")
        for model in failed:
            logger.info(f"  • {model}")
        sys.exit(1)

    logger.info(f"\nModels installed to: {models_dir}")


if __name__ == "__main__":
    main()
