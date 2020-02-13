import tensorflow.keras.backend as K
import tensorflow as tf


#This weights false positives and false negatives differently.
#Example:
#        model.compile(optimizer=opt,loss=mga_get_weighted_loss(np.array([[1.,20.]])),metrics=[f1,precision,recall,'binary_accuracy'])


def mga_get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(K.cast((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true)),dtype='float32')*K.cast(K.binary_crossentropy(K.cast(y_true,dtype='float32'), K.cast(y_pred,dtype='float32')),dtype='float32'), axis=-1)
    return weighted_loss

