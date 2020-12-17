
# summarze perception and attention branch network's
# both categorical cross entropy 
#


import tensorflow as tf
import numpy as np
from keras.losses import  mean_squared_error

class MultiLoss(object):
    
    """
    summarize both categorical cross entropy (prediction branch/ attention branch)
    """
    
    def __init__(self, num_classes):
        self.num_classes = num_classes


    def compute_loss(self, y_true, y_pred):
        """Compute mutlibox loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).

        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """

        cce = tf.keras.losses.CategoricalCrossentropy()
        loss_prediction_branch = cce(y_true[:4], y_pred[:4])

        cce_ = tf.keras.losses.CategoricalCrossentropy()
        loss_attention_branch = cce_(y_true[5:], y_pred[5:])

        total_loss = loss_prediction_branch + loss_attention_branch

        return total_loss

