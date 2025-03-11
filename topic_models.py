import random

random.seed(1234)

main_folder = 'main_path/'

src = main_folder + 'src'
raw_data = main_folder + 'raw_data'
tidy_data = main_folder + 'tidy_data'
experiments = main_folder + 'experiments'
coherence_folder = main_folder + 'experiments_coherence'

exec(open('libraries.py').read())

def dominant_topics(ldamodel = lda_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs):
  sent_topics_df = pd.DataFrame()
  
  for i, row in enumerate(ldamodel[corpus]):
    row = sorted(row, key = lambda x: (x[1]), reverse = True)
    
    for j, (topic_num, prop_topic) in enumerate(row):
      if j == 0:
        wp = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        sent_topics_df = sent_topics_df._append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index = True)
      else:
        break
  sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

  contents = pd.Series(texts)
  sent_topics_df = pd.concat([sent_topics_df, contents], axis = 1)
  
  return(sent_topics_df)

df1 = pd.read_json(os.path.join(tidy_data, 'bbc_preprocessed_0.01tfidf.json'))
df = df1.T[['body_clean']]
df = pd.DataFrame(df['body_clean'].str.join(', '))
df['topic'] = df1.T[['topic']]
df.to_csv(os.path.join(tidy_data, 'bbc_preprocessed_0.01tfidf_sheet.csv'), header = True, sep = '|', index_label = 'doc_id')
pd.read_csv(os.path.join(tidy_data, 'bbc_preprocessed_0.01tfidf_sheet.csv'), sep = '|', index_col = 0)

bbc = pd.read_csv(tidy_data + "/bbc_preprocessed_0.01tfidf_sheet.csv", sep = '|')
bbc['body_clean'] = bbc['body_clean'].apply(lambda x: x.split(','))
bbc_processed_docs = bbc['body_clean']
dictionary = Dictionary(bbc_processed_docs)
bbc_bow_corpus = [dictionary.doc2bow(doc) for doc in bbc_processed_docs]

lda_model = models.ldamodel.LdaModel(bbc_bow_corpus, num_topics = 5, id2word = dictionary, random_state = 100, chunksize = 1000, passes = 20, alpha = 0.5, eta = 0.1)

umass = CoherenceModel(model = lda_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'u_mass')
print("U-mass:" + str(umass.get_coherence())) 

cuci = CoherenceModel(model = lda_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_uci')
print("C-uci:" + str(cuci.get_coherence()))   

cnpmi = CoherenceModel(model = lda_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_npmi')
print("C-npmi:" + str(cnpmi.get_coherence())) 

df_topic_sents_keywords = dominant_topics(ldamodel = lda_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
  sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending = [0]).head(1)], axis = 0)

sent_topics_sorteddf_mallet.reset_index(drop = True, inplace = True)
sent_topics_sorteddf_mallet.columns = ['Topic_Number', "Contribution_Perc", "Keywords", "Text"]

bbc.loc[bbc['topic'] == 'business', 'topic'] = 'business'
bbc.loc[bbc['topic'] == 'politics', 'topic'] = 'politics'
bbc.loc[bbc['topic'] == 'sport', 'topic'] = 'sport'
bbc.loc[bbc['topic'] == 'technology', 'topic'] = 'technology'
bbc.loc[bbc['topic'] == 'entertainment', 'topic'] = 'entertainment'

df_lda_topic = pd.concat([bbc.reset_index(drop = True), df_dominant_topic['Dominant_Topic']], axis = 1)
df_lda_ct = pd.crosstab(df_lda_topic['topic'], df_lda_topic['Dominant_Topic'])
df_lda_ct.columns = ['sport', 'tech', 'business', 'entertainment', 'politics']
print(df_lda_ct)

mis_class_ws15 = df_lda_ct.reindex(sorted(df_lda_ct.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()

TP15 = np.diag(mis_class_ws15)
FP15 = np.sum(mis_class_ws15, axis = 0) - TP15
FN15 = np.sum(mis_class_ws15, axis = 1) - TP15

num_classes = 5
TN15 = []
for i in range(num_classes):
    temp = np.delete(mis_class_ws15, i, 0)
    temp = np.delete(temp, i, 1)
    TN15.append(sum(sum(temp)))

l = 2090

precision15 = TP15/(TP15 + FP15)
recall15 = TP15/(TP15 + FN15)
accuracy15 = (TP15 + TN15) / (TP15 + TN15 + FP15 + FN15)
f1_score15 = (2 * precision15 * recall15)/(precision15 + recall15)

topic = ['business', 'politics', 'sport', 'technology', 'entertainment']
measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())





nmf_model = models.Nmf(bbc_bow_corpus, num_topics = 5, id2word = dictionary, random_state = 100, chunksize = 1000, passes = 20)

umass = CoherenceModel(model = nmf_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'u_mass')
print("U-mass:" + str(umass.get_coherence())) 

cuci = CoherenceModel(model = nmf_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_uci')
print("C-uci:" + str(cuci.get_coherence()))   

cnpmi = CoherenceModel(model = nmf_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_npmi')
print("C-npmi:" + str(cnpmi.get_coherence()))

df_topic_sents_keywords = dominant_topics(ldamodel = nmf_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
  sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending = [0]).head(1)], axis = 0)

sent_topics_sorteddf_mallet.reset_index(drop = True, inplace = True)
sent_topics_sorteddf_mallet.columns = ['Topic_Number', "Contribution_Perc", "Keywords", "Text"]

bbc.loc[bbc['topic'] == 'business', 'topic'] = 'business'
bbc.loc[bbc['topic'] == 'politics', 'topic'] = 'politics'
bbc.loc[bbc['topic'] == 'sport', 'topic'] = 'sport'
bbc.loc[bbc['topic'] == 'technology', 'topic'] = 'technology'
bbc.loc[bbc['topic'] == 'entertainment', 'topic'] = 'entertainment'

df_lda_topic = pd.concat([bbc.reset_index(drop = True), df_dominant_topic['Dominant_Topic']], axis = 1)
df_lda_ct = pd.crosstab(df_lda_topic['topic'], df_lda_topic['Dominant_Topic'])
df_lda_ct.columns = ['politics', 'entertainment', 'tech', 'sport', 'business'] 
print(df_lda_ct)

mis_class_ws15 = df_lda_ct.reindex(sorted(df_lda_ct.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()

TP15 = np.diag(mis_class_ws15)
FP15 = np.sum(mis_class_ws15, axis = 0) - TP15
FN15 = np.sum(mis_class_ws15, axis = 1) - TP15

num_classes = 5
TN15 = []
for i in range(num_classes):
    temp = np.delete(mis_class_ws15, i, 0)
    temp = np.delete(temp, i, 1)
    TN15.append(sum(sum(temp)))

l = 2090

precision15 = TP15/(TP15 + FP15)
recall15 = TP15/(TP15 + FN15)
accuracy15 = (TP15 + TN15) / (TP15 + TN15 + FP15 + FN15)
f1_score15 = (2 * precision15 * recall15)/(precision15 + recall15)

topic = ['business', 'politics', 'sport', 'technology', 'entertainment']
measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())





lsi_model = models.LsiModel(bbc_bow_corpus, num_topics = 5, id2word = dictionary, chunksize = 1000)

umass = CoherenceModel(model = lsi_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'u_mass')
print("U-mass:" + str(umass.get_coherence())) 

cuci = CoherenceModel(model = lsi_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_uci')
print("C-uci:" + str(cuci.get_coherence()))   

cnpmi = CoherenceModel(model = lsi_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs, coherence = 'c_npmi')
print("C-npmi:" + str(cnpmi.get_coherence()))

df_topic_sents_keywords = dominant_topics(ldamodel = lsi_model, corpus = bbc_bow_corpus, texts = bbc_processed_docs)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
  sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending = [0]).head(1)], axis = 0)

sent_topics_sorteddf_mallet.reset_index(drop = True, inplace = True)
sent_topics_sorteddf_mallet.columns = ['Topic_Number', "Contribution_Perc", "Keywords", "Text"]

bbc.loc[bbc['topic'] == 'business', 'topic'] = 'business'
bbc.loc[bbc['topic'] == 'politics', 'topic'] = 'politics'
bbc.loc[bbc['topic'] == 'sport', 'topic'] = 'sport'
bbc.loc[bbc['topic'] == 'technology', 'topic'] = 'technology'
bbc.loc[bbc['topic'] == 'entertainment', 'topic'] = 'entertainment'

df_lda_topic = pd.concat([bbc.reset_index(drop = True), df_dominant_topic['Dominant_Topic']], axis = 1)
df_lda_ct = pd.crosstab(df_lda_topic['topic'], df_lda_topic['Dominant_Topic'])
df_lda_ct.columns = ['politics', 'entertainment', 'sport', 'tech', 'business'] 
print(df_lda_ct)

mis_class_ws15 = df_lda_ct.reindex(sorted(df_lda_ct.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()

TP15 = np.diag(mis_class_ws15)
FP15 = np.sum(mis_class_ws15, axis = 0) - TP15
FN15 = np.sum(mis_class_ws15, axis = 1) - TP15

num_classes = 5
TN15 = []
for i in range(num_classes):
    temp = np.delete(mis_class_ws15, i, 0)
    temp = np.delete(temp, i, 1)
    TN15.append(sum(sum(temp)))

l = 2090

precision15 = TP15/(TP15 + FP15)
recall15 = TP15/(TP15 + FN15)
accuracy15 = (TP15 + TN15) / (TP15 + TN15 + FP15 + FN15)
f1_score15 = (2 * precision15 * recall15)/(precision15 + recall15)

topic = ['business', 'politics', 'sport', 'technology', 'entertainment']
measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())





bbc_raw = pd.read_csv(raw_data + "/bbc_bert.csv")
docs_bbc = bbc_raw['text']
docs_bbc = docs_bbc.to_list()
y_bbc = bbc_raw['topic']
y_bbc = y_bbc.to_list()

y_bbc[:] = [0 if x == "entertainment" else x for x in y_bbc]
y_bbc[:] = [1 if x == "sport" else x for x in y_bbc]
y_bbc[:] = [2 if x == "politics" else x for x in y_bbc]
y_bbc[:] = [3 if x == "business" else x for x in y_bbc]
y_bbc[:] = [4 if x == "tech" else x for x in y_bbc]
y_bbc = np.array(y_bbc)

cluster_model = KMeans(n_clusters = 5)
empty_dimensionality_model = BaseDimensionalityReduction()

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sentence_model.encode(docs_bbc, show_progress_bar = True)
bt = BERTopic(calculate_probabilities = True, low_memory = False, nr_topics = 5, umap_model = empty_dimensionality_model, hdbscan_model = cluster_model)
topics, probs = bt.fit_transform(docs_bbc, embeddings)
bt.get_topic_info()

bbc_raw.loc[bbc_raw['topic'] == 'business', 'topic'] = 'business'
bbc_raw.loc[bbc_raw['topic'] == 'politics', 'topic'] = 'politics'
bbc_raw.loc[bbc_raw['topic'] == 'sport', 'topic'] = 'sport'
bbc_raw.loc[bbc_raw['topic'] == 'technology', 'topic'] = 'technology'
bbc_raw.loc[bbc_raw['topic'] == 'entertainment', 'topic'] = 'entertainment'

df = pd.DataFrame({"Document": docs_bbc, "Dominant_Topic": topics})
df_lda_topic = pd.concat([bbc_raw.reset_index(drop = True), df['Dominant_Topic']], axis = 1)
df_lda_ct = pd.crosstab(df_lda_topic['topic'], df_lda_topic['Dominant_Topic'])
df_lda_ct.columns = ['tech', 'sport', 'entertainment', 'politics', 'business']
print(df_lda_ct)

mis_class_ws15 = df_lda_ct.reindex(sorted(df_lda_ct.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()

TP15 = np.diag(mis_class_ws15)
FP15 = np.sum(mis_class_ws15, axis = 0) - TP15
FN15 = np.sum(mis_class_ws15, axis = 1) - TP15

num_classes = 5
TN15 = []
for i in range(num_classes):
    temp = np.delete(mis_class_ws15, i, 0)  
    temp = np.delete(temp, i, 1) 
    TN15.append(sum(sum(temp)))

l = 2090

precision15 = TP15/(TP15 + FP15)
recall15 = TP15/(TP15 + FN15)
accuracy15 = (TP15 + TN15) / (TP15 + TN15 + FP15 + FN15)
f1_score15 = (2 * precision15 * recall15)/(precision15 + recall15)

topic = ['politics', 'entertainment', 'sport', 'tech', 'business']
measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())

documents = pd.DataFrame({"Document": bbc_raw['text'], "ID": bbc_raw['doc_id'], "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index = False).agg({'Document': ' '.join})
cleaned_docs = bt._preprocess_text(documents_per_topic.Document.values)

vectorizer = bt.vectorizer_model
analyzer = vectorizer.build_analyzer()

words = vectorizer.get_feature_names()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in bt.get_topic(topic)] for topic in range(len(set(topics))-1)]

umass = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'u_mass')
print("U-mass:" + str(umass.get_coherence()))

cuci = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'c_uci')
print("C-uci:" + str(cuci.get_coherence()))

cnpmi = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'c_npmi')
print("C-npmi:" + str(cnpmi.get_coherence()))





bbc_raw = pd.read_csv(raw_data + "/bbc_bert.csv")
docs_bbc = bbc_raw['text']
docs_bbc = docs_bbc.to_list()
y_bbc = bbc_raw['topic']
y_bbc = y_bbc.to_list()

y_bbc[:] = [0 if x == "entertainment" else x for x in y_bbc]
y_bbc[:] = [1 if x == "sport" else x for x in y_bbc]
y_bbc[:] = [2 if x == "politics" else x for x in y_bbc]
y_bbc[:] = [3 if x == "business" else x for x in y_bbc]
y_bbc[:] = [4 if x == "tech" else x for x in y_bbc]
y_bbc = np.array(y_bbc)

cluster_model = KMeans(n_clusters = 5)
empty_dimensionality_model = BaseDimensionalityReduction()

sentence_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = sentence_model.encode(docs_bbc, show_progress_bar = True)
bt = BERTopic(calculate_probabilities = True, low_memory = False, nr_topics = 5, umap_model = empty_dimensionality_model, hdbscan_model = cluster_model)
topics, probs = bt.fit_transform(docs_bbc, embeddings)
bt.get_topic_info()

bbc_raw.loc[bbc_raw['topic'] == 'business', 'topic'] = 'business'
bbc_raw.loc[bbc_raw['topic'] == 'politics', 'topic'] = 'politics'
bbc_raw.loc[bbc_raw['topic'] == 'sport', 'topic'] = 'sport'
bbc_raw.loc[bbc_raw['topic'] == 'technology', 'topic'] = 'technology'
bbc_raw.loc[bbc_raw['topic'] == 'entertainment', 'topic'] = 'entertainment'

df = pd.DataFrame({"Document": docs_bbc, "Dominant_Topic": topics})
df_lda_topic = pd.concat([bbc_raw.reset_index(drop = True), df['Dominant_Topic']], axis = 1)
df_lda_ct = pd.crosstab(df_lda_topic['topic'], df_lda_topic['Dominant_Topic'])
df_lda_ct.columns = ['sport', 'business', 'entertainment', 'politics', 'tech']
print(df_lda_ct)

mis_class_ws15 = df_lda_ct.reindex(sorted(df_lda_ct.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()

TP15 = np.diag(mis_class_ws15)
FP15 = np.sum(mis_class_ws15, axis = 0) - TP15
FN15 = np.sum(mis_class_ws15, axis = 1) - TP15

num_classes = 5
TN15 = []
for i in range(num_classes):
    temp = np.delete(mis_class_ws15, i, 0)  
    temp = np.delete(temp, i, 1) 
    TN15.append(sum(sum(temp)))

l = 2090

precision15 = TP15/(TP15 + FP15)
recall15 = TP15/(TP15 + FN15)
accuracy15 = (TP15 + TN15) / (TP15 + TN15 + FP15 + FN15)
f1_score15 = (2 * precision15 * recall15)/(precision15 + recall15)

topic = ['politics', 'entertainment', 'sport', 'tech', 'business']
measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())

documents = pd.DataFrame({"Document": bbc_raw['text'], "ID": bbc_raw['doc_id'], "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index = False).agg({'Document': ' '.join})
cleaned_docs = bt._preprocess_text(documents_per_topic.Document.values)

vectorizer = bt.vectorizer_model
analyzer = vectorizer.build_analyzer()

words = vectorizer.get_feature_names()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in bt.get_topic(topic)] for topic in range(len(set(topics))-1)]

umass = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'u_mass')
print("U-mass:" + str(umass.get_coherence()))

cuci = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'c_uci')
print("C-uci:" + str(cuci.get_coherence()))

cnpmi = CoherenceModel(topics = topic_words, texts = tokens, corpus = corpus, dictionary = dictionary, coherence = 'c_npmi')
print("C-npmi:" + str(cnpmi.get_coherence()))