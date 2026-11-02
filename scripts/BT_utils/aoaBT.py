"""
Area of Applicability Analysis for Bradley-Terry Score Recovery
Decomposes domain shift by category: Consensus, Cultural Disagreement, 
Relational Ignorance, Structural Ignorance
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .evalBT import load_bt_scores_for_city, estimate_bt_scores_from_predictions

def compute_pairwise_bt_errors(test_data, true_bt_scores, estimated_bt_scores):
    """
    Compute BT prediction error for each pair
    
    For each pair (i, j), the BT model predicts:
        P(i > j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
    
    We compute:
        - True probability using true BT scores
        - Estimated probability using estimated BT scores
        - Absolute error between them
    
    Args:
        test_data: DataFrame with 'left' and 'right' columns
        true_bt_scores: dict {path: true_bt_score}
        estimated_bt_scores: dict {path: estimated_bt_score}
    
    Returns:
        array: Per-pair BT probability errors
    """
    
    errors = []
    
    for i, (idx, row) in enumerate(test_data.iterrows()):
        left_path = row['left']
        right_path = row['right']
        
        # Decode if bytes
        if isinstance(left_path, bytes):
            left_path = left_path.decode('utf-8')
        if isinstance(right_path, bytes):
            right_path = right_path.decode('utf-8')
        
        left_path = str(left_path)
        right_path = str(right_path)
        
        # Check if both locations have scores
        if left_path in true_bt_scores and right_path in true_bt_scores:
            if left_path in estimated_bt_scores and right_path in estimated_bt_scores:
                # True BT probability
                true_theta_l = true_bt_scores[left_path]
                true_theta_r = true_bt_scores[right_path]
                true_prob = np.exp(true_theta_l) / (np.exp(true_theta_l) + np.exp(true_theta_r))
                
                # Estimated BT probability
                est_theta_l = estimated_bt_scores[left_path]
                est_theta_r = estimated_bt_scores[right_path]
                est_prob = np.exp(est_theta_l) / (np.exp(est_theta_l) + np.exp(est_theta_r))
                
                # Absolute error
                error = abs(true_prob - est_prob)
                errors.append(error)
            else:
                errors.append(np.nan)
        else:
            errors.append(np.nan)
    
    return np.array(errors)

def compute_location_score_errors(true_bt_scores, estimated_bt_scores):
    """
    Compute per-location BT score errors
    
    Args:
        true_bt_scores: dict {path: true_score}
        estimated_bt_scores: dict {path: estimated_score}
    
    Returns:
        dict: {path: absolute_error}
    """
    
    common_locs = set(true_bt_scores.keys()) & set(estimated_bt_scores.keys())
    
    location_errors = {}
    for loc in common_locs:
        true_score = true_bt_scores[loc]
        est_score = estimated_bt_scores[loc]
        error = abs(true_score - est_score)
        location_errors[loc] = error
    
    return location_errors

def aggregate_bt_errors_by_category(test_data, pairwise_errors, categories):
    """
    Aggregate BT errors by AoA category
    
    Args:
        test_data: DataFrame with pairs
        pairwise_errors: Array of per-pair BT errors
        categories: Array of categories for each pair
    
    Returns:
        dict: Mean BT error per category
    """
    
    # Create a fresh DataFrame with sequential index
    df = pd.DataFrame({
        'bt_error': pairwise_errors,
        'category': categories
    })
    
    # Remove NaN errors
    df = df[~df['bt_error'].isna()]
    
    # Compute mean error per category
    category_errors = df.groupby('category')['bt_error'].agg(['mean', 'std', 'count'])
    
    return {
        'mean_errors': category_errors['mean'].to_dict(),
        'std_errors': category_errors['std'].to_dict(),
        'n_pairs': category_errors['count'].to_dict()
    }

def add_bt_metrics_to_categories(test_data, test_probs, categories_dict):
    """
    Add BT error metrics to existing category analysis
    
    Args:
        test_data: Test pairs DataFrame
        test_probs: Model predictions (probabilities)
        categories_dict: Dict from categorize_with_iso with category percentages
        target_city: Target city code
        bt_scores_db_path: Path to bt_scores.db
    
    Returns:
        dict: BT metrics for all 4 categories (consensus, cultural, relational, structural)
    """
    
    try:
        # Load true BT scores
        true_bt_scores = estimate_bt_scores_from_predictions(test_data, test_data['y'].values)
        
        # Estimate BT scores from model
        estimated_bt_scores = estimate_bt_scores_from_predictions(test_data, test_probs)
        
        # Compute pairwise BT errors
        pairwise_errors = compute_pairwise_bt_errors(
            test_data, true_bt_scores, estimated_bt_scores
        )
        
        # Get category labels (should be in categories_dict)
        if 'category_labels' not in categories_dict:
            return {
                'consensus_bt_error': None,
                'consensus_spearman': None,
                'cultural_bt_error': None,
                'cultural_spearman': None,
                'relational_bt_error': None,
                'relational_spearman': None,
                'structural_bt_error': None,
                'structural_spearman': None,
            }
        
        categories = categories_dict['category_labels']
        
        # Aggregate by category
        category_results = aggregate_bt_errors_by_category(
            test_data, pairwise_errors, categories
        )
        
        # Compute Spearman for all 4 categories
        category_spearman = {}
        
        all_categories = ['Consensus', 'Cultural Disagreement', 'Relational Ignorance', 'Structural Ignorance']
        
        for category in all_categories:
            # Get pairs in this category
            mask = np.array([c == category for c in categories]) & (~np.isnan(pairwise_errors))
            
            if mask.sum() < 2:
                category_spearman[category] = None
                continue
            
            # Get locations involved in these pairs
            category_pairs = test_data[mask]
            
            # Extract unique locations
            category_locations = set()
            for i, (idx, row) in enumerate(category_pairs.iterrows()):
                left = row['left']
                right = row['right']
                if isinstance(left, bytes):
                    left = left.decode('utf-8')
                if isinstance(right, bytes):
                    right = right.decode('utf-8')
                category_locations.add(str(left))
                category_locations.add(str(right))
            
            # Compute Spearman
            common_locs = category_locations & set(true_bt_scores.keys()) & set(estimated_bt_scores.keys())
            
            if len(common_locs) >= 2:
                true_vals = [true_bt_scores[loc] for loc in common_locs]
                est_vals = [estimated_bt_scores[loc] for loc in common_locs]
                corr, _ = spearmanr(true_vals, est_vals)
                category_spearman[category] = float(corr)
            else:
                category_spearman[category] = None
        
        # Return all 4 categories
        return {
            'consensus_bt_error': category_results['mean_errors'].get('Consensus'),
            'consensus_spearman': category_spearman.get('Consensus'),
            'cultural_bt_error': category_results['mean_errors'].get('Cultural Disagreement'),
            'cultural_spearman': category_spearman.get('Cultural Disagreement'),
            'relational_bt_error': category_results['mean_errors'].get('Relational Ignorance'),
            'relational_spearman': category_spearman.get('Relational Ignorance'),
            'structural_bt_error': category_results['mean_errors'].get('Structural Ignorance'),
            'structural_spearman': category_spearman.get('Structural Ignorance'),
        }
        
    except Exception as e:
        print(f"      Warning: BT category metrics failed - {e}")
        import traceback
        traceback.print_exc()
        return {
            'consensus_bt_error': None,
            'consensus_spearman': None,
            'cultural_bt_error': None,
            'cultural_spearman': None,
            'relational_bt_error': None,
            'relational_spearman': None,
            'structural_bt_error': None,
            'structural_spearman': None,
        }

def analyze_error_reduction_by_category(zero_shot_results, finetuned_results):
    """
    Compare BT errors before and after fine-tuning, by category
    
    Args:
        zero_shot_results: Results dict from decompose_bt_recovery_by_category (zero-shot)
        finetuned_results: Results dict from decompose_bt_recovery_by_category (fine-tuned)
    
    Returns:
        DataFrame: Error reduction analysis by category
    """
    
    categories = ['Consensus', 'Cultural Disagreement', 'Relational Ignorance', 'Structural Ignorance']
    
    comparison = []
    for cat in categories:
        zero_error = zero_shot_results['mean_bt_errors'].get(cat)
        fine_error = finetuned_results['mean_bt_errors'].get(cat)
        
        zero_spearman = zero_shot_results['category_spearman'].get(cat)
        fine_spearman = finetuned_results['category_spearman'].get(cat)
        
        if zero_error is not None and fine_error is not None:
            error_reduction = zero_error - fine_error
            pct_reduction = (error_reduction / zero_error) * 100
        else:
            error_reduction = None
            pct_reduction = None
        
        if zero_spearman is not None and fine_spearman is not None:
            spearman_improvement = fine_spearman - zero_spearman
        else:
            spearman_improvement = None
        
        comparison.append({
            'category': cat,
            'zero_shot_error': zero_error,
            'finetuned_error': fine_error,
            'error_reduction': error_reduction,
            'pct_reduction': pct_reduction,
            'zero_shot_spearman': zero_spearman,
            'finetuned_spearman': fine_spearman,
            'spearman_improvement': spearman_improvement
        })
    
    return pd.DataFrame(comparison)