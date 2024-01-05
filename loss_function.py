import tensorflow as tf

def logistic_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)  # Cast y_true to the same type as y_pred
    return tf.math.log1p(tf.math.exp(-y_true * y_pred))

def adaboost_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)  # Cast y_true to the same type as y_pred
    return tf.exp(-y_true * y_pred)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid taking the logarithm of zero
    y_true = tf.cast(y_true, y_pred.dtype)  # Cast y_true to the same type as y_pred
    loss = -y_true * tf.math.log(y_pred + epsilon) / tf.math.log(2.0) - \
           (1 - y_true) * tf.math.log(1 - y_pred + epsilon) / tf.math.log(2.0)
    return tf.reduce_mean(loss)