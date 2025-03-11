def louvain_function(path_graph, output_path_graph, matrix_edgelist, r):

    experiment_elements = {}
    community_edgelist_name = os.path.splitext(matrix_edgelist)[0]

    G = nx.read_weighted_edgelist(os.path.join(path_graph, matrix_edgelist), nodetype = str)
    G_louvain = algorithms.louvain(G, weight = 'weight', randomize = 1234, resolution = r)

    G_louvain2 = {}
    for idx, el in enumerate(G_louvain.communities):
        for j in el:
            G_louvain2.setdefault(j, idx)

    G_louvain2 = dict((el, G_louvain2[el]) for el in list(G.nodes))
    G_louvain_communities = np.unique(list(G_louvain2.values()))
    experiment_elements.setdefault(community_edgelist_name + '_louvain_number_communities', len(G_louvain_communities))

    G_louvain_com = []
    for term, comm in G_louvain2.items():
        G_louvain_com.append(comm)

    G_louvain_community = {}
    for k in G_louvain_com:
        comm_louvain = [key for key, value in G_louvain2.items() if value == k]
        G_louvain_community.setdefault('Community' + str(k + 1), ', '.join(comm_louvain))

    experiment_elements.setdefault(community_edgelist_name + '_louvain_community', G_louvain_community)

    readwrite.write_community_json(G_louvain, os.path.join(output_path_graph, community_edgelist_name + str(r) + '_louvain.json'))
    
    with open(os.path.join(output_path_graph, community_edgelist_name + str(r) + '_louvain_features.json'), 'w') as data:
        json.dump(experiment_elements, data)
        
    return None