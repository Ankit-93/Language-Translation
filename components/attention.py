import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Dense , Softmax 



class Encoder(tf.keras.Model):
 '''
 Encoder model -- That takes a input sequence and returns output sequence
 '''
 def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length):
 # Call the base class constructor
    super().__init__()

    # Initialize the variables
    self.vocabSize = inp_vocab_size
    self.embedDim = embedding_size
    self.lstmUnits = lstm_size//2 # Divide the LSTM Units by 2 because we would use Bidirectional LSTM with concatenation
    self.seqLength = input_length
    self.encOutput = 0

    self.forwardHState = 0 # Forward Hidden State
    self.forwardCState = 0 # Forward Cell State
    self.backwardHState = 0 # Backward Hidden State
    self.backwardCState = 0 # Backward Cell State
    self.hiddenState = 0 # Concatenated Hidden State
    self.cellState = 0 # Concatenated Cell State

    # Initialize Embedding layer
    self.embeddingLayer = layers.Embedding(input_dim=self.vocabSize, output_dim=self.embedDim, mask_zero=True,
    input_length=self.seqLength, name='embedding_layer_encoder')

    # Intialize Encoder LSTM layer
    self.lstmLayer = layers.LSTM(units=self.lstmUnits, return_sequences=True, return_state=True,
    name='lstm_layer_encoder')

    # Initialize Bidirectional Layer
    self.bidirectionLayer = layers.Bidirectional(self.lstmLayer)
    # Concatenate Layer
    self.concatLayer = layers.Concatenate()
 def call(self, input_sequence, states):
    '''
    This function takes a sequence input and the initial states of the encoder.
    Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
    returns -- All encoder_outputs, last time steps hidden and cell state
    '''
    embedOutput = self.embeddingLayer(input_sequence)

    self.encOutput, self.forwardHState, self.forwardCState, self.backwardHState, self.backwardCState = \
    self.bidirectionLayer(inputs=embedOutput, initial_state=states)

    # Concatenate forward and backward hidden states
    self.hiddenState = self.concatLayer([self.forwardHState, self.backwardHState])

    # Concatenate forward and backward cell states
    self.cellState = self.concatLayer([self.forwardCState, self.backwardCState])

    return self.encOutput, self.hiddenState, self.cellState

 def initialize_states(self, batch_size):
    '''
    Given a batch size it will return intial hidden state and intial cell state for both forward and backward pass
    of Bidirectional LSTM.
    If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
    '''

    return [tf.zeros(shape=(batch_size, self.lstmUnits)) for i in range(4)]


class Attention(tf.keras.layers.Layer):
  def __init__(self,scoring_function,att_units):
    super().__init__()
    self.scoring_function = scoring_function
    self.att_units = att_units

    if self.scoring_function=='dot':
      pass
    if scoring_function == 'general':
      self.W= tf.random.normal(shape=(self.att_units,self.att_units))
      pass
    elif scoring_function == 'concat':
      self.W1 =tf.random.normal(shape=(self.att_units,self.att_units))
      self.W2 =tf.random.normal(shape=(self.att_units,self.att_units))
      self.V  =tf.random.normal(shape=(self.att_units, 1))
      pass
  def call(self,state_hidden,encoder_output):
    att_weights=[]
    state_hidden = tf.expand_dims(state_hidden, axis = 2)
    att_unit = int(tf.shape(encoder_output)[2])
    if self.scoring_function=='dot':
      x = tf.matmul(encoder_output,state_hidden)
    elif self.scoring_function == 'general':
      x = tf.matmul(tf.tensordot(encoder_output, self.W , axes=1) ,state_hidden)
    elif self.scoring_function == 'concat':
      x = tf.tensordot(tf.tanh(tf.tensordot(encoder_output,self.W1,axes=1) + (tf.tensordot(tf.transpose(state_hidden,[0,2,1]),self.W2,axes=1))), self.V,axes=1)   
    xe = (Softmax()(tf.squeeze(x,axis=2)))
    vectors = tf.matmul(tf.transpose(encoder_output,[0,2,1]),K.expand_dims(xe, axis = 2))
    con_vec=(tf.squeeze(vectors,axis=2))
    return con_vec,tf.expand_dims(xe, axis = 2)

class One_Step_Decoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
   

      # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()

        self.tar_vocab_size=tar_vocab_size
        self.embedding_dim=embedding_dim
        self.input_length=input_length
        self.dec_units=dec_units
        self.score_fun=score_fun
        self.att_units=att_units
       
        self.embedding_osd= Embedding(input_dim = self.tar_vocab_size, output_dim = self.embedding_dim,
                                      input_length = self.input_length, name="embedding_layer_osd")
        self.lstm_osd= LSTM(self.dec_units, return_sequences=True,return_state=True,name="osd_LSTM")
       
        self.attention_osd = Attention(self.score_fun,self.att_units)
        self.dense_osd = Dense(self.tar_vocab_size)


  def call(self,input_to_decoder, encoder_output, state_h,state_c):

    embedding_op = self.embedding_osd(input_to_decoder)

    context_vector, weights = self.attention_osd(state_h,encoder_output)

    context_vector_new = tf.expand_dims(context_vector, axis = 1)

    concat_vector = tf.concat([embedding_op,tf.cast(context_vector_new, tf.float32)],axis = 2)

    osd_op,state_h_osd,state_c_osd= self.lstm_osd(concat_vector,initial_state =[state_h , state_c])
   
    final_op= (self.dense_osd(osd_op))

    final_op  = tf.squeeze(final_op, axis =1)
    return final_op,state_h_osd,state_c_osd,weights,context_vector

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        super(Decoder, self).__init__()
        self.out_vocab_size=out_vocab_size
        self.embedding_dim= embedding_dim
        self.input_length =input_length
        self.dec_units=dec_units
        self.score_fun=score_fun
        self.att_units=att_units
        self.onestep_decoder = One_Step_Decoder(self.out_vocab_size,self.embedding_dim,
                                          self.input_length,self.dec_units, self.score_fun,self.att_units)

       
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state ):

      
        all_output = tf.TensorArray(tf.float32,size=tf.shape(input_to_decoder)[1])

        for timestamp in range(tf.shape(input_to_decoder)[1]):
          if timestamp==0:
            decoder_hidden_state = encoder_output[:,-1]
          else:
            decoder_hidden_state = decoder_hidden_state

          final_op,decoder_hidden_state,decoder_cell_state,weights,context_vector=self.onestep_decoder(
              input_to_decoder[:,timestamp:timestamp+1],encoder_output,decoder_hidden_state,decoder_cell_state)
          all_output = all_output.write(timestamp,final_op)
  
        all_output = tf.transpose(all_output.stack(),[1,0,2])
        return all_output

class Encoder_decoder_attention(tf.keras.Model):
    
    def __init__(self,input_vocab_size, encoder_inputs_length,decoder_inputs_length, output_vocab_size,batch_size):
        super().__init__()
        self.batch_size= batch_size
        self.encoder = Encoder( inp_vocab_size=input_vocab_size+1, embedding_size=100 ,lstm_size=256, input_length=encoder_inputs_length)
        self.onestepdecoder = One_Step_Decoder(tar_vocab_size=output_vocab_size+1, embedding_dim=100, input_length=encoder_inputs_length, dec_units=256, score_fun='dot',att_units=256)
        self.dense = Dense(output_vocab_size+1) 
        self.decoder = Decoder(output_vocab_size+1, embedding_dim=100, input_length=20, dec_units=256,score_fun='dot' ,att_units=256)
    def call(self,data):
        enc_inp = data[0]
        dec_inp = data[1]
        initialEncState = self.encoder.initialize_states(self.batch_size)
        encoder_output, encoder_h, encoder_c = self.encoder(enc_inp, states=initialEncState)
        decoder_output = self.decoder(dec_inp,encoder_output, encoder_h, encoder_c)
        output = self.dense(decoder_output)


        return decoder_output

