import keras
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Dense , Softmax 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
import keras.backend as K
import warnings
warnings.filterwarnings('ignore')
################################################################
from preprocessing import *
from enc_dec import *
from dataloader import *
from callbacks import *
from attention import *
# importing the zipfile module
from zipfile import ZipFile
import os
import json

actual_path = os.getcwd()
print(actual_path)

data = get_data(actual_path)
data['english'] = data['english'].apply(preprocess)
data['italian'] = data['italian'].apply(preprocess_ita)

print("Input/Output example")
print("Italian sentence : %s" % data.loc[1000][1])
print("English sentence : %s" % data.loc[1000][0])
print()
print("Italian sentence : %s" % data.loc[2000][1])
print("English sentence : %s" % data.loc[2000][0])
print()
print("Italian sentence : %s" % data.loc[3000][1])
print("English sentence : %s" % data.loc[3000][0])
print()

inp_lengths = data['italian'].str.split().apply(len)
out_lengths = data['english'].str.split().apply(len)

def show_length_pencentile(range,language,length):
  for r in range:
    print('{}th percentile of {} sequence is :{}'.format(round(r, 1), language,round(np.percentile(length, r))))

input_maxlen = np.round(np.percentile(inp_lengths,99.9))
output_maxlen = np.round(np.percentile(out_lengths,99.9))

print('Maximum Sequence Length for Italian Language: ', input_maxlen)
print('Maximum Sequence Length for English Language: ', output_maxlen)


data['ita_lengths'] = data['italian'].str.split().apply(len)
data = data[data['ita_lengths'] < input_maxlen ]
data['eng_lengths'] = data['english'].str.split().apply(len)
data = data[data['eng_lengths'] < output_maxlen]

input_sentence = data['italian'].values
data['encoder_inp'] = data['italian'].astype(str)
data['english_inp'] = '<start> '+data['english'].astype(str)+' <end>'
output_sentence = data['english_inp'].values

train,test = train_test_split(data,test_size=0.1, random_state=4)
train,validation = train_test_split(train,test_size=0.1, random_state=4)
joblib.dump(test, "./dataset/test.pkl")

# joblib.load("model.pkl")
# with open('./dataset/test.pickle', 'wb') as handle:
#     pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(train.shape, validation.shape,test.shape)

train['decoder_inp'] = '<start> '+train['english'].astype(str)
train['decoder_out'] = train['english'].astype(str)+' <end>'

validation['decoder_inp'] = '<start> '+validation['english'].astype(str)
validation['decoder_out'] = validation['english'].astype(str)+' <end>'

ip_token = Tokenizer(filters='')
ip_token.fit_on_texts(input_sentence)

op_token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
op_token.fit_on_texts(output_sentence)

# joblib.dump(ip_token, "./dataset/ita_tokenizer.pickle")
# joblib.dump(op_token, "./dataset/eng_tokenizer.pickle")
ip_token_json = ip_token.to_json()
with open('./dataset/ita_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(ip_token_json, ensure_ascii=False))
op_token_json = op_token.to_json()
with open('./dataset/eng_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(op_token_json, ensure_ascii=False))
# with open('./dataset/ita_tokenizer.pickle', 'wb') as handle:
#     pickle.dump(ip_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./dataset/eng_tokenizer.pickle', 'wb') as handle:
#     pickle.dump(op_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_enc_inp = ip_token.texts_to_sequences(train['encoder_inp'])
train_dec_inp = op_token.texts_to_sequences(train['decoder_inp'])
train_dec_out = op_token.texts_to_sequences(train['decoder_out'])

val_enc_inp = ip_token.texts_to_sequences(validation['encoder_inp'])
val_dec_inp = op_token.texts_to_sequences(validation['decoder_inp'])
val_dec_out = op_token.texts_to_sequences(validation['decoder_out'])

train_encoder_seq = pad_sequences(train_enc_inp, padding='post')
train_decoder_inp_seq = pad_sequences(train_dec_inp, padding='post')
train_decoder_out_seq = pad_sequences(train_dec_out, padding='post')

val_encoder_seq = pad_sequences(val_enc_inp, padding='post')
val_decoder_inp_seq = pad_sequences(val_dec_inp, padding='post')
val_decoder_out_seq = pad_sequences(val_dec_out, padding='post')

input_vocab_size=len(ip_token.word_index)+1
output_vocab_size=len(op_token.word_index)+1

model = Encoder_Decoder(input_vocab_size,output_vocab_size)
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=["accuracy"])
train_steps=train.shape[0]//512
valid_steps=validation.shape[0]//512

model.fit([train_encoder_seq,train_decoder_inp_seq],train_decoder_out_seq, 
           validation_data = ([val_encoder_seq,val_decoder_inp_seq],val_decoder_out_seq), 
           steps_per_epoch = 400, 
           epochs=35,callbacks = [tensor, checkpoint,csv_logger])
model.summary()
model.save_weights('./attention/encDecModel.tf')
