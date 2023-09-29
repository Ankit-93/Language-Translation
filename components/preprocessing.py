import os
import re
import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

def preprocess(text):
    # convert all the text into lower letters
    # use this function to remove the contractions: https://gist.github.com/anandborad/d410a49a493b56dace4f814ab5325bbd
    # remove all the spacial characters: except space ' '
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text

def preprocess_ita(text):
    # convert all the text into lower letters
    # remove the words betweent brakets ()
    # remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
    # replace these spl characters with space: '\u200b', '\xa0', '-', '/'
    # we have found these characters after observing the data points, feel free to explore more and see if you can do find more
    # you are free to do more proprocessing
    # note that the model will learn better with better preprocessed data 
    text = str(text)
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text

def unzip(actual_path,datapath):
    print(datapath)
    with ZipFile(os.path.join(actual_path,datapath), 'r') as zObject:
        temp_path = os.path.join(actual_path,'dataset')
        os.chdir(temp_path)
        zObject.extractall()
        os.chdir(actual_path)

def get_data(actual_path):
    datapath = "./dataset/ita-eng.zip"
    unzip(actual_path,datapath)
    with open('./dataset/ita.txt' , 'r',encoding='utf-8') as f:
        eng=[]
        ita=[]
        for i in f.readlines():
            ita.append(i.split('\t')[1])
            eng.append(i.split('\t')[0])
    data = pd.DataFrame(data=list(zip(eng, ita)), columns=['english','italian'])
    os.remove("./dataset/ita.txt")
    return data

def get_embedding_matrix():
    actual_path=os.getcwd()
    with open('./dataset/ita_tokenizer.pickle', 'rb') as handle:
        ita_token = pickle.load(handle)
    with open('./dataset/eng_tokenizer.pickle', 'rb') as handle:
        eng_token = pickle.load(handle)
    output_vocab_size=len(eng_token.word_index)+1
    input_vocab_size=len(ita_token.word_index)+1
    #datapath = "sample_data/glove.6B.100d.zip"
    #unzip(actual_path,datapath)
    embeddings_index = dict()
    f = open('./dataset/glove.txt' , encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix_eng = np.zeros((output_vocab_size+1, 100))
    for word, i in eng_token.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_eng[i] = embedding_vector
    #os.remove("/content/NLP_Translation/dataset/glove.6B.100d.txt")
    return embedding_matrix_eng

