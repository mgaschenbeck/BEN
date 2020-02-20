import tensorflow.keras.backend as K
import tensorflow as tf


#This weights false positives and false negatives differently.
#Example:
#        model.compile(optimizer=opt,loss=mga_get_weighted_loss(np.array([[1.,20.]])),metrics=[f1,precision,recall,'binary_accuracy'])


def mga_get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(K.cast((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true)),dtype='float32')*K.cast(K.binary_crossentropy(K.cast(y_true,dtype='float32'), K.cast(y_pred,dtype='float32')),dtype='float32'), axis=-1)
    return weighted_loss

#the following is from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def mga_get_focal_loss(weights,alpha=0.25, gamma=2.0):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1.0 - y_pred) ** gamma * K.cast(targets,dtype='float32')
        weight_b = (1.0 - alpha) * y_pred ** gamma * (1 - K.cast(targets,dtype='float32'))
    
        return (tf.math.log1p(tf.math.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        loss = K.cast((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true)),dtype='float32')*loss

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss

def mga_macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost