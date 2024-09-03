from sklearn.metrics import log_loss
import numpy as np

def calculate_metrics(predictions, labels):
    # Apply sigmoid to get probabilities
    probabilities = 1 / (1 + np.exp(-predictions))
    
    # Calculate log loss
    loss = log_loss(labels, probabilities)
    
    return loss
