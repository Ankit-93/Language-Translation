import keras
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense , Softmax 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Layer
import keras.backend as K

################################################################
from preprocessing import *
from enc_dec import *
from dataloader import *
from callbacks import *
# importing the zipfile module
from zipfile import ZipFile
import os

actual_path = os.getcwd()
print(actual_path)

data = get_data(actual_path)
data['english'] = data['english'].apply(preprocess)
data['italian'] = data['italian'].apply(preprocess_ita)


ita_lengths= data['italian'].str.split().apply(len)
eng_lengths=data['english'].str.split().apply(len)

def show_length_pencentile(range,language,length):
  for r in range:
    print('{}th percentile of {} sequence is :{}'.format(round(r, 1), language,round(np.percentile(length, r))))

# show_length_pencentile(range(90, 101, 1),'English',eng_lengths)
# show_length_pencentile(np.arange(99, 100.1, 0.1),'English',eng_lengths)
# show_length_pencentile(np.arange(90, 101, 1),'Italian',ita_lengths)
# show_length_pencentile(np.arange(99, 100.1, 0.1),'Italian',ita_lengths)

print('Maximum Sequence Length for Italian Language: ', np.round(np.percentile(ita_lengths,99.9)))
print('Maximum Sequence Length for English Language: ', np.round(np.percentile(eng_lengths,99.9)))

df=data
df['ita_lengths'] = df['italian'].str.split().apply(len)
df = df[df['ita_lengths'] < 20 ]
df['eng_lengths'] = df['english'].str.split().apply(len)
df = df[df['eng_lengths'] < 20]
df.shape

final_data = df.drop(['eng_lengths','ita_lengths'],axis=1)
final_data = final_data

ita = final_data['italian'].values
english = final_data['english'].values



train,test = train_test_split(final_data,test_size=0.1, random_state=4)

train,validation = train_test_split(train,test_size=0.1, random_state=4)
print(train.shape, validation.shape,test.shape)

train['italian_inp'] =train['italian'].astype(str)
train['english_inp'] = '<start> '+train['english'].astype(str)
train['english_out'] = train['english'].astype(str)+' <end>'
train.sample(3)

validation['italian_inp'] =validation['italian'].astype(str)
validation['english_inp'] = '<start> '+validation['english'].astype(str)
validation['english_out'] = validation['english'].astype(str)+' <end>'
validation.sample(3)

train = train.drop(['english'],axis=1)
validation = validation.drop(['english'],axis=1)
train.head(2)

with open('./dataset/test.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

ita_token = Tokenizer(filters='')
ita_token.fit_on_texts(train['italian_inp'].values)

eng_token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
eng_token.fit_on_texts(train['english_inp'].values)
eng_token.fit_on_texts(train['english_out'].values)


with open('./dataset/ita_tokenizer.pickle', 'wb') as handle:
    pickle.dump(ita_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./dataset/eng_tokenizer.pickle', 'wb') as handle:
    pickle.dump(eng_token, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./dataset/ita_tokenizer.pickle', 'rb') as handle:
    ita_token = pickle.load(handle)
with open('./dataset/eng_tokenizer.pickle', 'rb') as handle:
    eng_token = pickle.load(handle)

ita_maxlen =np.round(np.percentile(ita_lengths,99.9))
eng_maxlen = np.round(np.percentile(eng_lengths,99.9))

output_vocab_size=len(eng_token.word_index)+1
input_vocab_size=len(ita_token.word_index)+1
train_dataset = Dataset(train, ita_token, eng_token, 20,20)
validation_dataset  = Dataset(validation, ita_token, eng_token, 20,20)

train_dataloader = Dataloder(train_dataset, batch_size=1024)
validation_dataloader = Dataloder(validation_dataset, batch_size=1024)

print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)

model  = Encoder_decoder(input_vocab_size=input_vocab_size, encoder_inputs_length=20,decoder_inputs_length=20,output_vocab_size=output_vocab_size)
if os.path.exists('./Attention'):
  model.load_weights('./Attention/encDecModel.tf')
else:
  pass

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy')
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024

model.fit(train_dataloader, validation_data = validation_dataloader, steps_per_epoch=train_steps, epochs=50,callbacks = [tensor, checkpoint])
model.summary()
model.save_weights('./Attention/encDecModel.tf')
