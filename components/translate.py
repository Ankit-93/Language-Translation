import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from nltk.translate import bleu_score
import nltk.translate.bleu_score as bleu
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from enc_dec import *

def translate(input_sentence):
  result = ''
  input_token = ita_token.texts_to_sequences([input_sentence])
  input_token = pad_sequences(input_token, maxlen=22,padding='post')
  enc_output, enc_state_h, enc_state_c = model.layers[0](input_token)
  decoder_input = np.reshape(eng_token.word_index['<start>'], (1,1))
  decoder_hidden_state = enc_state_h
  c = enc_state_c
  runLoop = True
  for i in range(25):  
    decoder_output,state_h,state_c = model.layers[1](decoder_input,states=[decoder_hidden_state,c])
    output = model.layers[2](decoder_output)    
    predicted_id = np.argmax(output[0])
    if eng_token.index_word[predicted_id] == '<end>':
        return result
    else:
        result += eng_token.index_word[predicted_id] + ' '
    decoder_input = tf.expand_dims([predicted_id], 0)
    decoder_hidden_state = state_h
    c = state_c
  return result


def translate_dataframe(data):
  length = len(data)
  index = data.index
  translated_sentence = []
  df = pd.DataFrame([], columns=['Italian', 'English (original)', 'English (translated)'])
  blues=0
  for i in index:
  
    reference= data.loc[i]['english']
    try:
      #print('###############',reference)  
      translation = translate(data.loc[i]['italian'])
      bleuscore = bleu.sentence_bleu([reference.split(),] , translation.split())
      blues = blues + bleuscore
      df_ = pd.DataFrame({'Italian':[data.loc[i]['italian']], 'English (original)':[ reference],
      'English (translated)': [translation]})
      df = pd.concat([df,df_])
      #print(translation ,':::',reference)
    except:
      pass

  avgBleuScore = blues/length

  return avgBleuScore,df

with open('./dataset/test.pickle', 'rb') as handle:
    test = pickle.load(handle)
with open('./dataset/ita_tokenizer.pickle', 'rb') as handle:
    ita_token = pickle.load(handle)
with open('./dataset/eng_tokenizer.pickle', 'rb') as handle:
    eng_token = pickle.load(handle)
output_vocab_size=len(eng_token.word_index)+1
input_vocab_size=len(ita_token.word_index)+1
model = Encoder_decoder(input_vocab_size=input_vocab_size, encoder_inputs_length=22,decoder_inputs_length=25,output_vocab_size=output_vocab_size)
model.load_weights("./encdecmodel/encDecModel.tf")
Bleuscore,dataframe = translate_dataframe(test[0:1000])
print('Average BleuScore: ', Bleuscore)
dataframe.to_excel("Translation.xlsx",index=False)