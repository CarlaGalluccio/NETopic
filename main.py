import random

random.seed(1234)

main_folder = '/Users/carlagalluccio/Desktop/Pubblicazioni/BDR/bbc/'

src = main_folder + 'src'
raw_data = main_folder + 'raw_data'
tidy_data = main_folder + 'tidy_data'
experiments = main_folder + 'experiments'
coherence_folder = main_folder + 'coherence_folder'

exec(open('libraries.py').read())

exec(open('text_cleaner_function.py').read())
exec(open('second_clean.py').read())
exec(open('co_occurrence_matrix_function.py').read())
exec(open('tfidf_reduction_function.py').read())
exec(open('co_matrix_to_edgelist_network_function.py').read())
exec(open('louvain_function.py').read())

stopwords_BBC = [line.rstrip('\n') for line in open(os.path.join(src, 'BBC.txt'), 'r', encoding = 'utf-8')]

data_to_clean = os.path.join(tidy_data, 'bbc.json')
text = 'all'
reduction_method = 'stemming'
n_gram = 1
num_topic = 5

text_cleaner(raw_data, tidy_data, 'bbc.json', text, reduction_method, n_gram, stopwords_BBC)
tfidf = [0.01, 0.1, 1, 0]

louv_param = [1, 1.07, 1.37, 1.50, 2]

for i in tfidf:
    for file_tfidf in os.listdir(tidy_data):
        if file_tfidf.endswith('preprocessed.json'):
            data_to_reduct = os.path.join(tidy_data, file_tfidf)
    tfidf_value = i
    tfidf_reduct(tidy_data, tidy_data, data_to_reduct, tfidf_value)

second_clean_file = []
for file in os.listdir(tidy_data):
    if file.endswith('tfidf.json'):
        second_clean_file.append(os.path.join(tidy_data, file))
        
for second_file in second_clean_file:
    second_clean(second_file)

type_co_occ = 'weighted'
window_size = [2, 5, 10, 15, 20]

for i in range(1, 31):
    experiment_folder = experiments + '/experiment' + str(i)
    
    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)
    os.makedirs(experiment_folder)
    
    
    
    for j in range(1, 101):
    	exp_path = experiment_folder + '/exp' + str(j)
    	
    	if os.path.exists(exp_path):
        	shutil.rmtree(exp_path)
    	os.makedirs(exp_path)
    
    j = 1
    for file in os.listdir(tidy_data):    
        for t in tfidf:
        	for w in window_size:
        		for r in louv_param:
        			exp_path = experiment_folder + '/exp' + str(j)
        			if file.startswith('bbc_preprocessed') and file.endswith('_' + str(t) + 'tfidf.json'):
        				co_occurrence_matrix(tidy_data, exp_path, file, type_co_occ, w)
        			
        				for col_index_file in os.listdir(exp_path):
        					if col_index_file.endswith('_matrix_columns.txt'):
        						col_index_name = [line.rstrip('\n') for line in open(os.path.join(exp_path, col_index_file), 'r', encoding = 'utf-8')]
        				
        				for co_file in os.listdir(exp_path):
        					if co_file.endswith('_matrix_sparse.npz'):
        						co_matrix_edgelist(exp_path, exp_path, co_file, col_index_name)
        					
        				for data_edgelist in os.listdir(exp_path):
        					if data_edgelist.endswith('edgelist.txt'):
        						louvain_function(exp_path, exp_path, data_edgelist, r)
        					
        						j = j + 1
