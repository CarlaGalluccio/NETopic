def text_cleaner(path_json, output_path_json_data, data_to_clean, text, reduction_method, n_gram, stopwords):
    
    json_name = os.path.splitext(data_to_clean)[0] + '_preprocessed'
    
    with open(os.path.join(path_json, data_to_clean), 'r') as file:
        data = json.load(file)

    for key, value in data.items():
        if text == 'body':
            text_to_clean = value['body']
        elif text == 'all':
            headline = value['headline']
            body = value['body']
            text_to_clean = headline + '. ' + body

        clean_text = re.sub(r'[\W+|\d]', ' ', text_to_clean)
        clean_text = re.sub(r'\b\w{1,2}\b', '', clean_text)
        clean_text = re.sub(r'(\s\W+)', ' ', clean_text)
        clean_text = re.sub(r'^\s+|\s+$', '', clean_text)

        lines_re_low = clean_text.lower().split()

        if n_gram == 1:
            lines_re_low = [word for word in lines_re_low if word not in stopwords]

        if reduction_method == 'stemming':
            clean_text_new = [' '.join([stemmer.stem(word) for word in lines_re_low])]
        elif reduction_method == 'lemmatisation':
            clean_text_new = [' '.join([token.lemma_ for sentence in lines_re_low for token in nlp_eng(sentence)])]

        clean_text_new_ts = [ngrams(word_tokenize(str(sentence)), n_gram) for sentence in clean_text_new] # text segmentation
        clean_text_new_ts = [' '.join(grams) for sentence in clean_text_new_ts for grams in sentence]
        
        value['body_clean'] = clean_text_new_ts
        
    with open(os.path.join(output_path_json_data, json_name), 'w') as data_cleaned:
        json.dump(data, data_cleaned)
        
    return data