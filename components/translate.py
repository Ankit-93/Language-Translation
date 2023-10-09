import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, Dense, Embedding, Dropout, Concatenate, Activation, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')
from nltk.translate import bleu_score
import nltk.translate.bleu_score as bleu
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from enc_dec import *
from preprocessing import *



def get_inference(model):
    encoder_input = model.input[0]
    _, state_h, state_c = model.layers[4].output
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_input, # input encoder model
                          encoder_states)# output encoder model

    # Decoder
    decoder_input = model.input[1]
    embeded_decoder = model.layers[3]
    embeded_decoder = embeded_decoder(decoder_input)
    decoder_state_h = Input(shape=(256), name="input_3")
    decoder_state_c = Input(shape=(256), name="input_4")
    decoder_state_inputs = [decoder_state_h,decoder_state_c]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h, state_c = decoder_lstm(embeded_decoder, initial_state=decoder_state_inputs)
    decoder_states = [state_h,state_c]

    # decoder outputs
    dropout_out = model.layers[6]
    dropout_out = dropout_out(decoder_outputs)
    decoder_dense = model.layers[7]
    decoder_outputs = decoder_dense(dropout_out)

    decoder_model = Model([decoder_input]+decoder_state_inputs, # input decoder model
                              [decoder_outputs]+decoder_states) 
    return encoder_model,decoder_model

def pre_processing_sentece(sentence):
    sentence = preprocess_ita(sentence)
    # Tokenize words
    sentence_tokenized = ip_token.texts_to_sequences([sentence])
    sentence_tokenized = pad_sequences(sentence_tokenized, padding="post")
    return sentence_tokenized

def inference_without_attention(sentence,op_token):
    encoder_model,decoder_model = get_inference(model)
    state_h, state_c = encoder_model.predict(sentence)
    target_word = np.zeros((1,1))
    target_word[0,0] = 1
    stop_condition=False
    sent=''
    step_size=0

    index_to_words = {idx: word for word, idx in op_token.word_index.items()}
    while not stop_condition:

        output, state_h, state_c = decoder_model.predict([target_word, state_h, state_c])
        output = np.argmax(output,-1)
        sent = sent+' '+str(index_to_words.get(int(output)))
        step_size+=1
        if step_size>20 or output==2:
          stop_condition = True
        target_word=output.reshape(1,1)

    return sent


def translate_dataframe(data,op_token):

    blues=0
    input_sentence = data['italian'].values
    output_sentence = data['english'].values
    translated=[]
    for ind,sent in enumerate(input_sentence):
        reference = output_sentence[ind]
        sentence_tokenized = pre_processing_sentece(sent)
        translated_sentence = inference_without_attention(sentence_tokenized,op_token)
        try:
          sentence_tokenized = str(sentence_tokenized).replace('<end>','')
        except:
          pass
        bleuscore = bleu.sentence_bleu([reference.split(),] , translated_sentence.split())
        blues = blues + bleuscore
        translated.append(translated_sentence)

    df = pd.DataFrame({'Italian':input_sentence, 'English (original)':output_sentence, 'English (translated)':translated})
    avgBleuScore = blues/len(input_sentence)
    print("The Average Bleu Score is: {}".format(avgBleuScore))
    return df

test = joblib.load("./dataset/test.pkl")
from tensorflow.keras.preprocessing.text import tokenizer_from_json
# ip_token = joblib.load("./dataset/ita_tokenizer.pkl")
# op_token = joblib.load("./dataset/eng_tokenizer.pkl")
with open('./dataset/ita_tokenizer.json') as f:
    data = json.load(f)
    ip_token = tokenizer_from_json(data)
with open('./dataset/eng_tokenizer.json') as f:
    data = json.load(f)
    op_token = tokenizer_from_json(data)

output_vocab_size=len(op_token.word_index)+1
input_vocab_size=len(ip_token.word_index)+1
model = Encoder_Decoder(input_vocab_size=input_vocab_size,output_vocab_size=output_vocab_size)
model.load_weights("./attention/encDecModel.tf")
# for layer in model.layers:
#     print(layer.name)
sentences = ["come stai?",
             "quanti anni hai?",
             "come ti chiami?",
             "è una bellissima giornata",
             "sei una ragazza pericolosa",
             "ho studiato duramente per superare l'esame",
             "mia madre dice sempre che sono bello"
             ]

for sentence in sentences:
    sentence_tokenized = pre_processing_sentece(sentence)
    translated_sentence = inference_without_attention(sentence_tokenized,op_token)
    print("Input  sentence :  %s" % sentence)
    print("Output sentence : %s" % translated_sentence)
    print()

dataframe = translate_dataframe(test.iloc[810:820],op_token)
dataframe.to_excel("Translation.xlsx",index=False)
