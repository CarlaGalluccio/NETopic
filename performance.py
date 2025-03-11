import random

random.seed(1234)

main_folder = 'main_path/'

src = main_folder + 'src'
raw_data = main_folder + 'raw_data'
tidy_data = main_folder + 'tidy_data'
experiments = main_folder + 'experiments'
coherence_folder = main_folder + 'coherence_folder'

exec(open('libraries.py').read())

with open(os.path.join(tidy_data, 'bbc_preprocessed_0.01tfidf.json'), 'r') as file:
    data_red = json.load(file)
    
topic = ['sport', 'tech', 'business', 'entertainment', 'politics']

right_topic = {}
graph_element = {}

for key, value in data_red.items():
    right_topic.setdefault(value['doc_id'], value['topic'])
    
df_right_topic = pd.DataFrame(right_topic.items(), columns = ['Document', 'Observed'])

graph_value = {}
g_val = []
g_key = []
topic_temp = []
graph_value_topic = []

topic_value = {}
graph_topic_community = {}
n = 1
j = 1

with open(os.path.join(experiments, 'experiment1/exp21', 'bbc0.01tfidf_co_ws20_matrix_sparse_edgelist1_louvain_features.json'), 'r') as file:
    louvain_ws20 = json.load(file)
    
for key, value in louvain_ws20[list(louvain_ws20.keys())[1]].items():
    g_key.append(key)
    g_val.append(''.join(value).split(', '))

for i in range(len(g_key)):
    graph_value.setdefault(g_key[i], g_val[i])

for ky, vl in data_red.items():
    topic_value.setdefault(vl['doc_id'], vl['body_clean'])
    
for el in graph_value.values():
    for word in topic_value.values():
        graph_list = list(set(el).intersection(word))
        graph_value_topic.append(graph_list)
    graph_topic_community.setdefault('C' + str(j), list(graph_value_topic))
    graph_value_topic = []
    j += 1
    
df_graph_topic_community = pd.DataFrame()
df_graph_topic_community_new = pd.DataFrame()
df_graph_topic_community_perc = pd.DataFrame()
    
for m in range(len(louvain_ws20[list(louvain_ws20.keys())[1]].items())):
    df_graph_topic_community['C' + str(m + 1)] = graph_topic_community.get('C' + str(m + 1))
    df_graph_topic_community_new['C' + str(m + 1)] = df_graph_topic_community['C' + str(m + 1)].str.len()
        
df_graph_topic_community_new = df_graph_topic_community_new.set_axis(list(topic_value.keys()), axis = 0)
df_graph_topic_community = df_graph_topic_community.set_axis(list(topic_value.keys()), axis = 0)

df_find_topic = pd.DataFrame(df_graph_topic_community_new.idxmax(axis = 1), columns = ['Predicted'])
df_find_topic = df_find_topic.rename_axis('Document').reset_index()

df_topic_table = df_right_topic.merge(df_find_topic, on = 'Document')
    
df_topic_table2 = df_topic_table.pivot_table(index = 'Observed', columns = 'Predicted', values = 'Document', aggfunc = 'size', fill_value = 0)
    
graph_element.setdefault('Mclass_table_ws20_louvain', df_topic_table2)
print(df_topic_table2)

mis_class_ws15 = graph_element['Mclass_table_ws20_louvain']
mis_class_ws15.columns = ['sport', 'business', 'entertainment', 'politics', 'tech']
print(mis_class_ws15)

mis_class_ws15 = mis_class_ws15.reindex(sorted(mis_class_ws15.columns), axis = 1)
mis_class_ws15 = mis_class_ws15.to_numpy()
print(mis_class_ws15)

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

topic = ['sport', 'business', 'entertainment', 'politics', 'tech']

measure_ws20_louvain = pd.DataFrame({'Topic': topic, 'Precision': list(precision15), 'Recall': list(recall15), 'F1-Score': list(f1_score15), 'Accuracy': list(accuracy15)})
print(measure_ws20_louvain)

print(measure_ws20_louvain[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean())