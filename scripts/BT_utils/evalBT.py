"""
Bradley-Terry Score Evaluation Utilities
Functions to compare model-estimated BT scores to true pre-computed scores
"""

import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.linear_model import LogisticRegression

def load_bt_scores_for_city(city, bt_scores_db_path):
    """
    Load pre-computed BT scores for a city
    
    Args:
        city: City code (e.g., 'NBO', 'ACC')
        bt_scores_db_path: Path to bt_scores.db
    
    Returns:
        dict: {path: bt_score}
    """
    con = sqlite3.connect(bt_scores_db_path)
    query = f"SELECT path, bt_score FROM bt_scores WHERE city = '{city}'"
    df = pd.read_sql(query, con=con)
    con.close()
    
    scores = dict(zip(df['path'], df['bt_score']))
    return scores

def estimate_bt_scores_from_predictions(pairs_df, predictions):
    """
    Estimate BT scores from model predictions
    
    Args:
        pairs_df: DataFrame with 'left' and 'right' columns
        predictions: Array of predicted probabilities P(left > right)
    
    Returns:
        dict: {location_path: estimated_bt_score}
    """
    
    # Get unique locations
    all_left = pairs_df['left'].values
    all_right = pairs_df['right'].values
    
    # Decode bytes if needed
    all_left = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in all_left]
    all_right = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in all_right]
    
    unique_locations = sorted(list(set(all_left) | set(all_right)))
    n_locs = len(unique_locations)
    
    # Create location index mapping
    loc_to_idx = {loc: i for i, loc in enumerate(unique_locations)}
    
    # Build feature matrix
    X = []
    y = []
    
    for i, (idx, row) in enumerate(pairs_df.iterrows()):
        left_path = row['left']
        right_path = row['right']
        
        # Decode if bytes
        if isinstance(left_path, bytes):
            left_path = left_path.decode('utf-8')
        if isinstance(right_path, bytes):
            right_path = right_path.decode('utf-8')
        
        left_path = str(left_path)
        right_path = str(right_path)
        
        # Create feature vector
        features = np.zeros(n_locs)
        features[loc_to_idx[left_path]] = 1
        features[loc_to_idx[right_path]] = -1
        
        X.append(features)
        
        # Use model prediction as "vote"
        # Convert probability to binary (>0.5 means left wins)
        y.append(1 if predictions[i] > 0.5 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Fit BT model from predictions
    model = LogisticRegression(
        fit_intercept=False,
        max_iter=1000,
        solver='lbfgs'
    )
    
    model.fit(X, y)
    
    # Extract estimated BT scores
    bt_scores = {}
    for loc, idx in loc_to_idx.items():
        bt_scores[loc] = float(model.coef_[0][idx])
    
    # Normalize (mean = 0)
    mean_score = np.mean(list(bt_scores.values()))
    bt_scores = {loc: score - mean_score for loc, score in bt_scores.items()}
    
    return bt_scores

def evaluate_bt_score_recovery(true_scores, estimated_scores):
    """
    Compare true BT scores to model-estimated scores
    
    Args:
        true_scores: dict {location: true_bt_score}
        estimated_scores: dict {location: estimated_bt_score}
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Get locations present in both
    common_locs = set(true_scores.keys()) & set(estimated_scores.keys())
    
    if len(common_locs) < 2:
        # Not enough overlap to compute correlations
        return {
            'spearman': None,
            'kendall': None,
            'pearson': None,
            'score_rmse': None,
            'score_mae': None,
            'top_10_overlap': None,
            'n_locations': len(common_locs)
        }
    
    # Extract scores in same order
    locations = sorted(list(common_locs))
    true_vals = np.array([true_scores[loc] for loc in locations])
    est_vals = np.array([estimated_scores[loc] for loc in locations])
    
    # Ranking correlations
    spearman_corr, _ = spearmanr(true_vals, est_vals)
    kendall_corr, _ = kendalltau(true_vals, est_vals)
    
    # Score correlations
    pearson_corr, _ = pearsonr(true_vals, est_vals)
    
    # Score errors
    score_rmse = np.sqrt(np.mean((true_vals - est_vals)**2))
    score_mae = np.mean(np.abs(true_vals - est_vals))
    
    # Ranking accuracy (top-10 overlap)
    if len(locations) >= 10:
        true_ranking = np.argsort(true_vals)[::-1]  # Descending
        est_ranking = np.argsort(est_vals)[::-1]
        
        true_top10 = set(true_ranking[:10])
        est_top10 = set(est_ranking[:10])
        
        top_10_overlap = len(true_top10 & est_top10) / 10.0
    else:
        top_10_overlap = None
    
    return {
        'spearman': float(spearman_corr),
        'kendall': float(kendall_corr),
        'pearson': float(pearson_corr),
        'score_rmse': float(score_rmse),
        'score_mae': float(score_mae),
        'top_10_overlap': float(top_10_overlap) if top_10_overlap is not None else None,
        'n_locations': len(common_locs)
    }

# def evaluate_model_bt_recovery(model, test_data, test_preds, target_city, bt_scores_db_path, 
#                                  root_path, sub_bands):
def evaluate_model_bt_recovery(test_data, test_preds):
    """
    Complete BT score recovery evaluation for a model
    
    Args:
        model: Fine-tuned model
        test_data: DataFrame with test pairs
        test_preds: Model predictions (probabilities)
        target_city: Target city code
        bt_scores_db_path: Path to BT scores database
        root_path: Root data path
        sub_bands: Sub-bands for images
    
    Returns:
        dict: BT evaluation metrics
    """
    
    try:
        # Load true BT scores
        true_scores = estimate_bt_scores_from_predictions(test_data, test_data['y'].values)
        
        # Estimate BT scores from model predictions
        estimated_scores = estimate_bt_scores_from_predictions(test_data, test_preds)
        
        # Compare
        metrics = evaluate_bt_score_recovery(true_scores, estimated_scores)
        
        return metrics
        
    except Exception as e:
        print(f"      Warning: BT evaluation failed - {e}")
        return {
            'spearman': None,
            'kendall': None,
            'pearson': None,
            'score_rmse': None,
            'score_mae': None,
            'top_10_overlap': None,
            'n_locations': 0
        }