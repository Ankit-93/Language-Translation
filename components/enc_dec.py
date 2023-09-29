import numpy as np
import tensorflow as tf
from preprocessing import get_embedding_matrix
from tensorflow.keras.layers import Embedding, LSTM, Dense , Softmax

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, lstm_size,input_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.lstm_size= lstm_size
        self.lstm_output = 0
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    def call(self, input_sentances, training=True ):
        input_embedd = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c

    def initialize_states(self,batch_size):
        self.lstm_state_h=np.zeros(shape=(batch_size, self.lstm_size))
        self.lstm_state_c=np.zeros(shape=(batch_size,self.lstm_size))
        return self.lstm_state_h, self.lstm_state_c

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units,input_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.input_length = input_length
        embedding_matrix_eng = get_embedding_matrix()
        # we are using embedding_matrix and not training the embedding layer
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, weights = [embedding_matrix_eng],name="embedding_layer_decoder",trainable=False)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Decoder_LSTM")
       
    def call(self,target_sentances, states):
        target_embedd = self.embedding(target_sentances)
        lstm_output,hidden_state,cell_state = self.lstm(target_embedd, initial_state=states)
        return lstm_output,hidden_state,cell_state

class Encoder_decoder(tf.keras.Model):
    
    def __init__(self,input_vocab_size, encoder_inputs_length,decoder_inputs_length, output_vocab_size,batch_size=1024):
        super().__init__()
        self.encoder = Encoder(vocab_size=input_vocab_size+1, embedding_dim=50, input_length=encoder_inputs_length, lstm_size=256)
        self.decoder = Decoder(vocab_size=output_vocab_size+1, embedding_dim=100,input_length=decoder_inputs_length, dec_units=256)
        self.dense = Dense(output_vocab_size , activation='softmax')
        self.batch_size = batch_size 
    def call(self,data):
        enc_inp = data[0]
        dec_inp = data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(enc_inp )
        initial_state = [encoder_h, encoder_c]
        decoder_output,_,_ = self.decoder(dec_inp,initial_state)
        output = self.dense(decoder_output)
        return output

