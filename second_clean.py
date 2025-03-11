def second_clean(data_to_clean):
    
    with open(os.path.join(data_to_clean), 'r') as file:
        data = json.load(file)
    
    second_clean_name = os.path.basename(os.path.splitext(data_to_clean)[0])
    
    bbc_newdict = {}
    body_list = []

    for key, value in data.items():
      body_text = value['body']
      if body_text not in body_list:
        bbc_newdict[key] = value
        body_list.append(body_text)

    new_word_list = []

    for key, value in bbc_newdict.items():
        new_word_list.extend(value['body_clean'])

    counts = Counter(new_word_list)

    uncommon_words = []

    for word, count in counts.items():
      if count == 1:
        uncommon_words.append(word)

    for key, value in bbc_newdict.items():
      value['body_clean'] = [word for word in value['body_clean'] if word not in uncommon_words]

    with open(os.path.join(tidy_data, second_clean_name + '.json'), 'w') as data_tfidf_file:
      json.dump(bbc_newdict, data_tfidf_file)
      
    return None