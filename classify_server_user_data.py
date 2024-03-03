import argparse
import os

from app.classification import classify

SDF_TRAIN_PATH = '/davidb/guyshur/kompot/data/kegg/data_dir/sdf_train.pkl'
TRAIN_EMBEDDINGS_PATH = '/davidb/guyshur/kompot/data/kegg/data_dir/embeddings_train.pkl'
MODEL_PATH = '/davidb/guyshur/kompot/data/kegg/data_dir/model.pt'

def main(data_dir_path):
    output_path = os.path.join(data_dir_path, "predictions.tsv")
    classify(
        sdf_train_path=SDF_TRAIN_PATH,
        sdf_classify_path=os.path.join(data_dir_path, "sdf_classify.pkl"),
        model_path=MODEL_PATH,
        train_embeddings_path=TRAIN_EMBEDDINGS_PATH,
        classification_embeddings_path="",
        output_path=output_path,
        device='cuda',
        threshold=0.0734,
        chunksize=20_000,
        nthreads=40,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_path",
        type=str,
        help="[REQUIRED] Path to input data directory",
        required=True,
    )

    args = parser.parse_args()
    data_dir_path = args.data_dir_path
    main(data_dir_path)
