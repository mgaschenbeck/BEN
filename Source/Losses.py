import tensorflow.keras.backend as K
import tensorflow as tf
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(K.round(y_true), K.round(y_pred)), axis=-1)
    return weighted_loss