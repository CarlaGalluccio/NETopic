import random

random.seed(1234)

main_folder = 'main_path/'

src = main_folder + 'src'
raw_data = main_folder + 'raw_data'
tidy_data = main_folder + 'tidy_data'
experiments = main_folder + 'experiments'
coherence_folder = main_folder + 'coherence_folder'

exec(open('libraries.py').read())



for i in range(1, 31):
	experiments_new = experiments + '/experiment' + str(i)
	
	for j in range(21, 22):
		experiment_folder = experiments_new + '/exp' + str(j)
		
		community_features = []
		community_edgelist = []
		
		for data_community in os.listdir(experiment_folder):
			if data_community.endswith('louvain_features.json'):
				community_features.append(os.path.join(experiment_folder, data_community))
			if data_community.endswith('edgelist.txt'):
				community_edgelist.append(os.path.join(experiment_folder, data_community))
				
		with open(community_features[0], 'r') as file:
			bbc_louvain_features = json.load(file)
		G_bbc = nx.read_weighted_edgelist(community_edgelist[0], nodetype = str)
		
		communities_key = []
		communities_value = []
		
		for key, value in bbc_louvain_features[list(bbc_louvain_features.keys())[1]].items():
			communities_key.append(key)
			communities_value.append(''.join(value).split(', '))
			
		subgraph_element = {}
		for k in range(len(communities_key)):
			community_graph = G_bbc.subgraph(communities_value[k])
			subgraph_element.setdefault(communities_key[k], community_graph)
			
		c = 1
		community_dict = {}
		
		for key, value in subgraph_element.items():
			degree = nx.degree_centrality(subgraph_element['Community' + str(c)])
		
			co_data = pd.DataFrame.from_records([degree], index = ['degree']).T
			co_data = co_data.reset_index(level = 0)
			
			co_data_dict = co_data.to_dict('dict')
			community_dict.setdefault('Community' + str(c), co_data_dict)
			
			co_data.to_csv(os.path.join(experiment_folder, 'exp' + str(j) + '_community' + str(c) + '.csv'), sep = '\t', index = False)
			
			c += 1
			
		with open(os.path.join(experiment_folder, 'exp' + str(j) + '_communities_measures.json'), 'w') as data:
			json.dump(community_dict, data)



df_ci_final = pd.DataFrame(columns = ['U-mass','C-uci','C_npmi'])

for exp_gen in range(1, 31):
	experiments_new = experiments + '/experiment' + str(exp_gen)
	
	folder = list(range(21, 22))
		
	with open(os.path.join(tidy_data, 'bbc_preprocessed_0.01tfidf.json'), 'r') as file:
		data_bbc = json.load(file)

	bbc_texts = []

	for value in data_bbc.values():
		bbc_texts.append(value['body_clean'])
		
	dictionary_bbc = corpora.Dictionary(bbc_texts)
	corpus_bbc = [dictionary_bbc.doc2bow(text) for text in bbc_texts]
	
	communities_list = []
	experiments_folder = []
	
	for f in folder:
		exp = experiments_new + '/exp' + str(f)
		experiments_folder.append(exp)
		
	for idx, element in enumerate(folder):
		for data_community in os.listdir(experiments_folder[idx]):
			if data_community.endswith('_measures.json'):
				communities_list.append(data_community)
				
	tmp_dict = {}
	ci_dict = {}
	
	df_ci = pd.DataFrame(columns = ['U-mass','C-uci','C_npmi'])
	
	for idx, element in enumerate(folder):
		with open(os.path.join(experiments_folder[idx], communities_list[idx]), 'r') as file:
			data = json.load(file)
			
		for k, v in data.items():
			idx = v['index']
			dci = v['degree']
		
			for i, j in idx.items():
				for m, n in dci.items():
					if i == m:
						tmp_dict.setdefault(j, n)
					
			ci_dict.setdefault(k, tmp_dict)
			tmp_dict = {}
		
			bbc_topics = []
		
			for key, value in ci_dict.items():
				bbc_comm_list = sorted(value, key = value.__getitem__, reverse = True)
				bbc_topics.append(bbc_comm_list[:500])
			
			umass = CoherenceModel(topics = bbc_topics, corpus = corpus_bbc, texts = bbc_texts, dictionary = dictionary_bbc, coherence = 'u_mass')
			cuci = CoherenceModel(topics = bbc_topics, corpus = corpus_bbc, texts = bbc_texts, dictionary = dictionary_bbc, coherence = 'c_uci')
			cnpmi = CoherenceModel(topics = bbc_topics, corpus = corpus_bbc, texts = bbc_texts, dictionary = dictionary_bbc, coherence = 'c_npmi')
			
			df_row = {'U-mass': umass.get_coherence(), 'C-uci': cuci.get_coherence(), 'C_npmi': cnpmi.get_coherence()}
			df_ci = df_ci._append(df_row, ignore_index = True)

	df_ci_final_csv = df_ci.reset_index(level = 0)
	df_ci_final_csv.to_csv(os.path.join(coherence_folder, 'coherence_experiment' + str(exp_gen) + '_node_coherence.csv'), sep = '\t', index = False)
    
	df_ci_final = pd.DataFrame(columns = ['U-mass','C-uci','C_npmi'])