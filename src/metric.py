from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def calculate_metrics(predictions, labels):
    # Apply sigmoid to get probabilities
    predictions = 1 / (1 + np.exp(-predictions))
    # Convert probabilities to binary predictions
    predictions = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': accuracy
    }