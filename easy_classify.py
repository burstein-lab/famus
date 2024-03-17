import argparse

from app.classification import classify
from app import get_cfg


if __name__ == "__main__":
    cfg = get_cfg()
    default_device = cfg["device"]
    default_chunksize = cfg["chunksize"]
    default_threshold = cfg["threshold"]
    default_nthreads = cfg["nthreads"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf_train_path",
        type=str,
        help="Path to the sdf file used for training the model",
    )
    parser.add_argument(
        "--sdf_classify_path",
        type=str,
        help="Path to the sdf file used for classification",
    )
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument(
        "--train_embeddings_path", type=str, help="Path to the embeddings file"
    )
    parser.add_argument(
        "--classification_embeddings_path",
        type=str,
        help="Path to the embeddings file",
    )
    parser.add_argument("--output_path", type=str, help="Path to the output file")
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use for classification",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=default_chunksize,
        help="Number of sequences to classify at once",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=default_threshold,
        help="Threshold for classification",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=default_nthreads,
        help="Number of threads to use for classification",
    )

    args = parser.parse_args()
    classify(
        sdf_train_path=args.sdf_train_path,
        sdf_classify_path=args.sdf_classify_path,
        model_path=args.model_path,
        train_embeddings_path=args.train_embeddings_path,
        classification_embeddings_path=args.classification_embeddings_path,
        output_path=args.output_path,
        device=args.device,
        chunksize=args.chunksize,
        threshold=args.threshold,
        nthreads=args.nthreads,
    )
