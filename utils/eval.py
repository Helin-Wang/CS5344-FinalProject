import numpy as np
from scipy.special import erf
from sklearn.metrics import roc_auc_score, average_precision_score

def decision_function_to_probability(scores, method='minmax'):
    """
    Convert decision function scores to probabilities using multiple methods.
    
    Args:
        scores: Raw decision function scores from IsolationForest
        method: 'minmax', 'zscore', 'quantile', or 'robust'
        
    Returns:
        probabilities: Probabilities in range [0, 1] where higher values indicate higher anomaly probability
    """
    scores = np.array(scores)
    
    if method == 'minmax':
        # Min-max normalization
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores)
        probabilities = (scores - min_score) / (max_score - min_score)
        
    elif method == 'zscore':
        # Z-score normalization with erf
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score == 0:
            return np.zeros_like(scores)
        z_scores = (scores - mean_score) / std_score
        probabilities = 0.5 * (1 + erf(z_scores / np.sqrt(2)))
        
    elif method == 'quantile':
        # Quantile-based normalization
        q25, q75 = np.percentile(scores, [25, 75])
        if q75 == q25:
            return np.zeros_like(scores)
        probabilities = np.clip((scores - q25) / (q75 - q25), 0, 1)
        
    elif method == 'robust':
        # Robust normalization using median and MAD
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))
        if mad == 0:
            return np.zeros_like(scores)
        probabilities = 1 / (1 + np.exp(-(scores - median_score) / (1.4826 * mad)))
        
    return probabilities

def evaluate_scores(estimator, X, y_true, k=0.1, prob_method='minmax'):
    """
    Evaluate anomaly detection model using probability-based metrics.
    - Converts decision_function scores to probabilities using multiple methods
    - Higher probabilities indicate higher anomaly likelihood
    """
    scores = estimator.decision_function(X)
    
    # For IsolationForest, lower decision_function scores indicate anomalies
    # So we invert the scores first
    inverted_scores = -scores
    
    # Convert decision function scores to probabilities
    anomaly_probabilities = decision_function_to_probability(inverted_scores, method=prob_method)
        
    # Calculate multiple metrics using probabilities
    roc = roc_auc_score(y_true, anomaly_probabilities)
    ap = average_precision_score(y_true, anomaly_probabilities)
        
    # Precision@k
    idx = np.argsort(anomaly_probabilities)[::-1]
    top_k_count = int(len(y_true) * k)
    top_k = y_true[idx][:top_k_count]
    precision_at_k = np.mean(top_k)
    
    return anomaly_probabilities, roc, ap, precision_at_k