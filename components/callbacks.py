import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    why are we using this, can't we use simple sparse categorical crossentropy?
    Yes, you can use simple sparse categorical crossentropy as loss like we did in task-1. But in this loss function we are ignoring the loss
    for the padded zeros. i.e when the input is zero then we donot need to worry what the output is. This padded zeros are added from our end
    during preprocessing to make equal length for all the sentences.

    """
    
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

from keras.callbacks import ModelCheckpoint
filepath = 'callbacks/EncDecModel.tf'
checkpoint = ModelCheckpoint(filepath = filepath, monitor ='val_loss' ,mode='min',save_weights_only=True, save_best_only = True , verbose=0)


from keras.callbacks import TensorBoard,CSVLogger
tensor = TensorBoard(log_dir='logs',histogram_freq=1,write_graph=True,write_grads=True)

csv_logger = CSVLogger("Keras.log", separator=",", append=True)