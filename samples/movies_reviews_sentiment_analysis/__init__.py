import os
import pandas as pd

BASE_DIR = "./data/movie_reviews/"

def get_labeled_train_data(base_dir=BASE_DIR):
    return pd.read_csv(os.path.join(base_dir, "labeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)

def get_test_unlabeled_data(base_dir=BASE_DIR):
    return pd.read_csv(os.path.join(base_dir, "testData.tsv"), header=0, delimiter="\t", quoting=3)

def get_unlabeled_train_data(base_dir=BASE_DIR):
    return pd.read_csv(os.path.join(base_dir, "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)
