def count_dict(text_to_reduct, word_set):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in text_to_reduct:
            if word in sent:
                word_count[word] += 1
    return word_count


def tfidf_reduct(path_data, output_path_data, data_to_reduct, tfidf_value):
    text_to_reduct = []
    word_tf_idf = []
    clean_text_new_tfidf = []
    
    with open(os.path.join(path_data, data_to_reduct), 'r') as file:
        data = json.load(file)

    for key, value in data.items():
        text_to_reduct.append(value['body_clean'])
        
    word_set = set(np.unique(flatten(text_to_reduct)))
    total_documents = len(text_to_reduct)
    word_count = count_dict(text_to_reduct, word_set)

    for sentence in text_to_reduct:
        N = len(sentence)
        for word in sentence:
            occurrence = len([token for token in sentence if token == word])
            tf = occurrence/N

            word_occurrence = word_count[word]
            idf = np.log(total_documents/word_occurrence)

            value = tf*idf
            word_tf_idf.append((word, value))

    df_tfidf = pd.DataFrame(word_tf_idf, columns = ['word', 'tf-idf'])
    df_tfidf_reduct = df_tfidf.groupby(['word']).sum().reset_index()
    words = (df_tfidf_reduct.sort_values('tf-idf', ascending = False))
    data_tfidf = dict(words.values)

    for value in data.values():
        for el in value['body_clean']:
            for tf_term, tfidf in data_tfidf.items():
                if el == tf_term and tfidf > tfidf_value:
                    clean_text_new_tfidf.append(el)
                        
        value['body_clean'] = clean_text_new_tfidf
        clean_text_new_tfidf = []
            
    words.to_csv(os.path.join(output_path_data, os.path.splitext(data_to_reduct)[0] + '_tfidf_values.txt'), sep = '\t', index = False, mode = 'w')
    
    with open(os.path.join(output_path_data, os.path.splitext(data_to_reduct)[0] + '_' + str(tfidf_value) + 'tfidf.json'), 'w') as data_tfidf:
        json.dump(data, data_tfidf)

    return data