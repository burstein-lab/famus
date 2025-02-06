from app.classification import classify
from easy_preprocess_classify import main as easy_process


input_path = "/davidb/guyshur/high_res_ko_project/test_sequences.fasta"
input_full_profiles_dir_path = "data/kegg/data_dir/subcluster_profiles/"
input_sdf_train_path = "data/kegg/data_dir/sdf_train.pkl"
data_dir_path = "data/time_test/"
model_path = "data/kegg/data_dir/model.pt"
embeddings_train_path = "data/kegg/data_dir/embeddings_train.pkl"
easy_process(
    input_fasta_file_path=input_path,
    input_full_profiles_dir_path=input_full_profiles_dir_path,
    input_sdf_train_path=input_sdf_train_path,
    data_dir_path=data_dir_path,
    n_processes=40,
)
classify(
    sdf_train_path=input_sdf_train_path,
    sdf_classify_path=data_dir_path + "sdf_classify.pkl",
    model_path=model_path,
    train_embeddings_path=embeddings_train_path,
    classification_embeddings_path=data_dir_path + "cls_emb.pkl",
    output_path=data_dir_path + "output.tsv",
    device="cuda",
    threshold=0.0734,
    n_processes=40,
)
