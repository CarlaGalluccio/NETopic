def co_matrix_edgelist(path_matrix, output_path_matrix, matrix_edgelist_network, col_index_name):

    co_data_npz = sparse.load_npz(os.path.join(path_matrix, matrix_edgelist_network))
    co_data_to_edgelist = pd.DataFrame.sparse.from_spmatrix(co_data_npz, columns = col_index_name, index = col_index_name)
    matrix_edgelist_network_name = os.path.splitext(matrix_edgelist_network)[0]
    
    co_data_to_edgelist = co_data_to_edgelist.reindex(co_data_to_edgelist.columns)
    g_data = nx.from_pandas_adjacency(co_data_to_edgelist, create_using = nx.Graph())
    data_edge = nx.to_pandas_edgelist(g_data)
    data_edge.to_csv(os.path.join(output_path_matrix, matrix_edgelist_network_name + '_edgelist.txt'), header = False, index = False, sep = '\t')

    return None