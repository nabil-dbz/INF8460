import pandas as pd

def read_ann_file(path):
    lines = open(path, "r", encoding='utf-8').readlines()
    result = [line.strip().split("\t") for line in lines if not line[0] != "T"]
    result = [[row[0]] + row[1].split(" ") + [row[2]] for row in result]
    df = pd.DataFrame(result)
    df.rename(
        columns={
            0: 'Ti', 
            1: 'Type', 
            2: 'Span_Min', 
            3: 'Span_Max', 
            4: 'Word'
        },
        inplace=True
    )
    return df

def get_tag(words, index, key_word_type):
    t = '' if len(key_word_type) == 0 else '-' + key_word_type[0]
    if len(words) == 1:
        return 'U' + t
    if index == 0:
        return 'B' + t
    if index == len(words) - 1:
        return 'L' + t
    return 'I' + t

def get_tags_from_ann_structure(path, doc_id, should_consider_type):
    tags = dict()
    ann_df = read_ann_file(path + doc_id + '.ann')
    for row_index in range(len(ann_df)):
        key_word = ann_df.loc[row_index]['Word'].split(' ')
        min_index = int(ann_df.loc[row_index]['Span_Min'])
        key_word_type = ann_df.loc[row_index]['Type'] if should_consider_type else ''
        index_key_part = 0
        for index_key_part in range(len(key_word)):
            part = key_word[index_key_part]
            max_index = min_index + len(part)
            if (part, min_index, max_index) not in tags or tags[(part, min_index, max_index)][0] != 'I':
                tags[(part, min_index, max_index)] = get_tag(key_word, index_key_part, key_word_type)
            min_index = max_index + 1
    return tags

def complete_document_tags(path, doc_id, data, should_consider_type):
    tokens = data['Token'].loc[data['DocID'] == doc_id].to_list()
    tags_from_ann = get_tags_from_ann_structure(path, doc_id, should_consider_type)
    text = open(path + doc_id + '.txt', "r", encoding='utf-8').read().replace('Â', '')
    index = 0
    token_id = 0
    for token_id in range(len(tokens)):
        token = str(tokens[token_id])
        min_index = text[index:].find(token) + index
        max_index = min_index + len(token)
        if (token, min_index, max_index) not in tags_from_ann:
            data['Tag'].loc[data['TokenID'] == doc_id + '-' + str(token_id)] = 'O'
        else:
            data['Tag'].loc[data['TokenID'] == doc_id + '-' + str(token_id)] = tags_from_ann[(token, min_index, max_index)] 
        index = max_index
    return data
    
def complete_BILOU(data, should_consider_type, dataset, path):
    print('Strating to fill in the ', dataset, ' dataframe with the right tags...')
    doc_ids = set(data['DocID'].to_list())
    new_data = data.copy()
    counter = 0
    for doc_id in doc_ids:
        new_data = complete_document_tags(path + dataset + '/' + dataset + '/', doc_id, new_data, should_consider_type)
        counter += 1
        if counter % 20 == 0:
            print(counter, 'documents have been processed...')
    print('Done!')
    return new_data