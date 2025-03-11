def co_occurrence_matrix(path_json, output_path, json_text_clean, type_co_occ, window_size):
    d = collections.defaultdict(int)
    vocab = set()
    m = 0
    
    with open(os.path.join(path_json, json_text_clean), 'r') as line:
            data = json.load(line)
        
    for key, value in data.items():
        text = value['body_clean']

        for i in range(len(text)):
            token = text[i]
            vocab.add(token)
            if isinstance(window_size, int):
                next_token = text[i + 1 : i + 1 + window_size]
            elif window_size is None:
                next_token = text[i + 1 : i + 1 + len(text)]

            for t in next_token:
                key = tuple(sorted([t, token]))
                if type_co_occ == 'binary':
                    d[key] = 1
                elif type_co_occ == 'weighted':
                    d[key] += 1
                elif type_co_occ == 'proximity':
                    if isinstance(window_size, int):
                        if m < (window_size + 1):
                            d[key] += round((window_size - m)/window_size, 2)
                            m += 1
                    elif window_size is None:
                        if m < (len(text) + 1):
                            d[key] += round((len(text) - m)/len(text), 2)
                            m += 1
            m = 0

    vocab = sorted(vocab)
    df = pd.DataFrame(data = np.zeros((len(vocab), len(vocab))), index = vocab, columns = vocab)

    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    
    json_text_clean = re.sub('.json', '', json_text_clean)
    co_matrix_name = json_text_clean.split('_')[0] + json_text_clean.split('_')[2] + '_co_ws' + str(window_size)
    
    pd.Series(df.index).to_csv(os.path.join(output_path, co_matrix_name + '_matrix_columns.txt'), header = False, index = False)
    df_scipy = scipy.sparse.csr_matrix(df.values)
    sparse.save_npz(os.path.join(output_path, co_matrix_name + '_matrix_sparse.npz'), df_scipy)
                
    return df