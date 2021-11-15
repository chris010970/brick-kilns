import keras
import numpy as np

import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(y_true, y_pred):
    
    """
    Custom loss function
    """
    
    # y_true and y_pred are of the form [prob_pos,prob_neg]
    y_true = tf.reshape(y_true, [-1,2])
    y_pred = tf.reshape(y_pred, [-1,2])
 
    # compute weights
    weights = tf.math.reduce_sum(1 - y_true, axis=0)
    loss = K.binary_crossentropy(y_true, y_pred)
  
    return K.mean(loss*weights)


def recall_m(y_true, y_pred):

    """
    recall
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):

    """
    precision
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):

    """
    f1 score
    """

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""
def mean_iou( y_true, y_pred ):

    Custom loss function

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
"""

def dice_coef(  y_true, y_pred ):

    """
    Custom loss function
    """

    # 
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss( y_true, y_pred ):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss( y_true, y_pred ):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def compute_dice(im1, im2, empty_score=1.0):

    """
    Evaluation metric: Dice
    """

    im1 = np.asarray(im1>0.5).astype(np.bool)
    im2 = np.asarray(im2>0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
