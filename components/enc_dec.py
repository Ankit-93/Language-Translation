import numpy as np
import tensorflow as tf
from preprocessing import get_embedding_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense , Softmax
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, Dense, Embedding, Dropout, Concatenate, Activation, Dot


def Encoder_Decoder(input_vocab_size,output_vocab_size):
    encoder_input = Input(shape=[None],dtype=tf.int32)
    encoder_embedding = Embedding(input_dim=input_vocab_size+1,
                        output_dim=256,
                        mask_zero=True)(encoder_input)
    _, enc_state_h, enc_state_c = LSTM(256,
                                        return_state=True,
                                        return_sequences=False)(encoder_embedding)
    # Save the state of the last step of the encoder which will be the initial state of the decoder.
    encoder_state = [enc_state_h, enc_state_c]  

    # Decoder
    decoder_input = Input(shape=[None], dtype=tf.int32)
    decoder_embedding = Embedding(input_dim=output_vocab_size+1,
                                    output_dim=256,
                                    mask_zero=True)(decoder_input)
    decoder_lstm_output, dec_state_h, dec_state_c = LSTM(256, 
                                                        return_sequences=True,
                                                        return_state=True)(decoder_embedding, initial_state=encoder_state)
            
    decoder_lstm_output_dropout = Dropout(0.5)(decoder_lstm_output)
    decoder_output = TimeDistributed(Dense(output_vocab_size, activation="softmax"))(decoder_lstm_output_dropout)
    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])
    return model
    


