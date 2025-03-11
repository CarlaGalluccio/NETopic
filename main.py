import os
import random
import shutil
random.seed(1234)

main_folder = 'main_path'
folders = {
    "src": "src",
    "raw_data": "raw_data",
    "tidy_data": "tidy_data",
    "experiments": "experiments",
    "coherence_folder": "coherence_folder"
}
folders = {key: os.path.join(main_folder, val) for key, val in folders.items()}

from libraries import *
from text_cleaner_function import text_cleaner
from second_clean import second_clean
from co_occurrence_matrix_function import co_occurrence_matrix
from tfidf_reduction_function import tfidf_reduct
from co_matrix_to_edgelist_network_function import co_matrix_edgelist
from louvain_function import louvain_function

stopwords_BBC = [line.strip() for line in open(os.path.join(folders["src"], 'BBC.txt'), encoding='utf-8')]

text = 'all'
reduction_method = 'stemming'
n_gram = 1
num_topic = 5
tfidf_values = [0.01, 0.1, 1, 0]
louv_params = [1, 1.07, 1.37, 1.50, 2]
window_sizes = [2, 5, 10, 15, 20]
type_co_occ = 'weighted'

text_cleaner(folders["raw_data"], folders["tidy_data"], 'bbc.json', text, reduction_method, n_gram, stopwords_BBC)

for tfidf_value in tfidf_values:
    for file_tfidf in os.listdir(folders["tidy_data"]):
        if file_tfidf.endswith('preprocessed.json'):
            data_to_reduct = os.path.join(folders["tidy_data"], file_tfidf)
            tfidf_reduct(folders["tidy_data"], folders["tidy_data"], data_to_reduct, tfidf_value)

second_clean_files = [os.path.join(folders["tidy_data"], f) for f in os.listdir(folders["tidy_data"]) if f.endswith('tfidf.json')]
for file in second_clean_files:
    second_clean(file)

for i in range(1, 31):
    experiment_folder = os.path.join(folders["experiments"], f'experiment{i}')
    shutil.rmtree(experiment_folder, ignore_errors=True)
    os.makedirs(experiment_folder)
    
    for j in range(1, 101):
        exp_path = os.path.join(experiment_folder, f'exp{j}')
        os.makedirs(exp_path, exist_ok=True)
    
    j = 1
    for file in os.listdir(folders["tidy_data"]):
        if file.startswith('bbc_preprocessed') and file.endswith('tfidf.json'):
            for tfidf_value in tfidf_values:
                if file.endswith(f'_{tfidf_value}tfidf.json'):
                    for w in window_sizes:
                        for r in louv_params:
                            exp_path = os.path.join(experiment_folder, f'exp{j}')
                            co_occurrence_matrix(folders["tidy_data"], exp_path, file, type_co_occ, w)
                            
                            col_index_name = []
                            for col_index_file in os.listdir(exp_path):
                                if col_index_file.endswith('_matrix_columns.txt'):
                                    with open(os.path.join(exp_path, col_index_file), encoding='utf-8') as f:
                                        col_index_name = [line.strip() for line in f]
                            
                            for co_file in os.listdir(exp_path):
                                if co_file.endswith('_matrix_sparse.npz'):
                                    co_matrix_edgelist(exp_path, exp_path, co_file, col_index_name)
                            
                            for data_edgelist in os.listdir(exp_path):
                                if data_edgelist.endswith('edgelist.txt'):
                                    louvain_function(exp_path, exp_path, data_edgelist, r)
                            
                            j += 1

