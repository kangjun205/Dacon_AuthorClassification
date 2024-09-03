from sklearn.metrics import log_loss
import numpy as np

def calculate_metrics(predictions, labels):
    # Apply softmax to get probabilities for multi-class
    exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculate log loss
    loss = log_loss(labels, probabilities)
    
    return loss
