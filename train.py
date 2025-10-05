from utils import make_preprocessing_pipeline, split_dataset, evaluate_scores, decision_function_to_probability   
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import joblib
from itertools import product
from typing import Dict, Any, Tuple, Optional, Callable
import warnings

# Predefined parameter grids for common anomaly detection models
DEFAULT_PARAM_GRIDS = {
    'IsolationForest': {
        "n_estimators": [100, 200, 300, 500],
        "max_samples": ["auto", 256, 512, 1024],
        "max_features": [0.5, 0.7, 0.9, 1.0],
        "contamination": [0.08, 0.1, 0.12, 0.15, 0.18],
        "bootstrap": [True, False]
    },
    'OneClassSVM': {
        "kernel": ['rbf', 'poly', 'sigmoid'],
        "gamma": ['scale', 'auto', 0.1, 0.01],
        "nu": [0.05, 0.1, 0.15, 0.2, 0.25]
    },
    'LocalOutlierFactor': {
        "n_neighbors": [10, 20, 30, 50],
        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "contamination": [0.08, 0.1, 0.12, 0.15, 0.18]
    }
}

def get_default_param_grid(model_name: str) -> Dict[str, Any]:
    """
    Get default parameter grid for a given model.
    
    Args:
        model_name: Name of the model ('IsolationForest', 'OneClassSVM', 'LocalOutlierFactor')
        
    Returns:
        Dictionary containing parameter grid
    """
    if model_name not in DEFAULT_PARAM_GRIDS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_PARAM_GRIDS.keys())}")
    return DEFAULT_PARAM_GRIDS[model_name]

def get_model_class(model_name: str):
    """
    Get model class for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class
    """
    model_classes = {
        'IsolationForest': IsolationForest,
        'OneClassSVM': OneClassSVM,
        'LocalOutlierFactor': LocalOutlierFactor
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")
    return model_classes[model_name]

def train_anomaly_detection_model(
    model_class,
    param_grid: Dict[str, Any],
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    preprocessing_pipeline: Optional[Callable] = None,
    scoring_weights: Dict[str, float] = None,
    probability_methods: list = None,
    save_model_path: str = None,
    save_submission_path: str = None,
    verbose: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generic training workflow for anomaly detection models.
    
    Args:
        model_class: The model class to instantiate (e.g., IsolationForest)
        param_grid: Dictionary of parameter grids for hyperparameter tuning
        train_data_path: Path to training data CSV
        val_data_path: Path to validation data CSV  
        test_data_path: Path to test data CSV
        preprocessing_pipeline: Function that returns preprocessing pipeline (default: make_preprocessing_pipeline)
        scoring_weights: Weights for composite scoring (default: {'pr_auc': 0.6, 'roc_auc': 0.2, 'precision_at_k': 0.2})
        probability_methods: List of probability conversion methods to test (default: ['minmax', 'zscore', 'quantile', 'robust'])
        save_model_path: Path to save the best model (default: 'results/best_model.pkl')
        save_submission_path: Path to save submission file (default: 'results/submission.csv')
        verbose: Whether to print progress information
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing:
        - 'best_model': The best trained model
        - 'best_params': Best parameters found
        - 'best_score': Best validation score
        - 'results_df': DataFrame with all parameter combinations and scores
        - 'test_predictions': Test set anomaly probabilities
        - 'validation_metrics': Final validation metrics
    """
    
    # Set defaults
    if preprocessing_pipeline is None:
        preprocessing_pipeline = make_preprocessing_pipeline
    if scoring_weights is None:
        scoring_weights = {'pr_auc': 0.6, 'roc_auc': 0.2, 'precision_at_k': 0.2}
    if probability_methods is None:
        probability_methods = ['minmax', 'zscore', 'quantile', 'robust']
    if save_model_path is None:
        save_model_path = 'results/best_model.pkl'
    if save_submission_path is None:
        save_submission_path = 'results/submission.csv'
    
    if verbose:
        print("="*60)
        print("ANOMALY DETECTION MODEL TRAINING")
        print("="*60)
        print(f"Model: {model_class.__name__}")
        print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Load and preprocess data
    if verbose:
        print("\nLoading and preprocessing data...")
    
    df_train = pd.read_csv(train_data_path)
    df_val = pd.read_csv(val_data_path)
    df_test = pd.read_csv(test_data_path)
    
    X_train, y_train = split_dataset(df_train)
    X_val, y_val = split_dataset(df_val)
    
    preprocess = preprocessing_pipeline()
    X_train_processed = preprocess.fit_transform(X_train)
    X_val_processed = preprocess.transform(X_val)
    X_test_processed = preprocess.transform(df_test)
    
    if verbose:
        print(f"Training data shape: {X_train_processed.shape}")
        print(f"Validation data shape: {X_val_processed.shape}")
        print(f"Test data shape: {X_test_processed.shape}")
        print(f"Training anomaly rate: {y_train.mean():.4f}")
        print(f"Validation anomaly rate: {y_val.mean():.4f}")
    
    # Hyperparameter tuning
    if verbose:
        print(f"\nStarting hyperparameter tuning...")
    
    results = []
    best_model = None
    best_params = None
    best_score = -np.inf
    
    all_keys = list(param_grid.keys())
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    for i, values in enumerate(product(*param_grid.values())):
        params = dict(zip(all_keys, values))
        
        try:
            # Create and fit model
            model = model_class(random_state=random_state, n_jobs=-1, **params)
            model.fit(X_train_processed)
            
            # Evaluate on validation set
            val_scores, val_roc, val_ap, val_precision_at_k = evaluate_scores(
                model, X_val_processed, y_val
            )
            
            # Calculate composite score
            composite_score = (val_ap * scoring_weights['pr_auc'] + 
                             val_roc * scoring_weights['roc_auc'] + 
                             val_precision_at_k * scoring_weights['precision_at_k'])
            
            results.append({
                **params,
                "val_roc_auc": val_roc,
                "val_pr_auc": val_ap,
                "val_precision_at_k": val_precision_at_k,
                "composite_score": composite_score
            })
            
            # Update best model
            if composite_score > best_score:
                best_score = composite_score
                best_model = model
                best_params = params
                if verbose:
                    print(f"New best score: {best_score:.4f} (iteration {i+1})")
                    print(f"  ROC-AUC: {val_roc:.4f}, PR-AUC: {val_ap:.4f}, Precision@k: {val_precision_at_k:.4f}")
            
            # Progress reporting
            if verbose and ((i + 1) % 50 == 0 or (i + 1) == total_combinations):
                print(f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) - Best: {best_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"Error with params {params}: {e}")
            continue
    
    if verbose:
        print(f"\nHyperparameter tuning completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
    
    # Final validation evaluation
    final_val_scores, final_val_roc, final_val_ap, final_val_precision_at_k = evaluate_scores(
        best_model, X_val_processed, y_val
    )
    
    validation_metrics = {
        'roc_auc': final_val_roc,
        'pr_auc': final_val_ap,
        'precision_at_k': final_val_precision_at_k,
        'composite_score': best_score
    }
    
    if verbose:
        print(f"\nFinal validation metrics:")
        for metric, value in validation_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Test set predictions with best probability conversion method
    if verbose:
        print(f"\nTesting probability conversion methods...")
    
    val_scores_raw = best_model.decision_function(X_val_processed)
    best_prob_method = None
    best_prob_score = -np.inf
    
    for method in probability_methods:
        try:
            val_probs = decision_function_to_probability(-val_scores_raw, method=method)
            val_roc = roc_auc_score(y_val, val_probs)
            val_ap = average_precision_score(y_val, val_probs)
            method_score = val_ap * 0.7 + val_roc * 0.3
            
            if verbose:
                print(f"Method '{method}': ROC={val_roc:.4f}, PR={val_ap:.4f}, Score={method_score:.4f}")
            
            if method_score > best_prob_score:
                best_prob_score = method_score
                best_prob_method = method
        except Exception as e:
            if verbose:
                print(f"Method '{method}' failed: {e}")
            continue
    
    if verbose:
        print(f"Best probability conversion method: {best_prob_method}")
    
    # Generate test predictions
    test_scores_raw = best_model.decision_function(X_test_processed)
    test_predictions = decision_function_to_probability(-test_scores_raw, method=best_prob_method)
    
    if verbose:
        print(f"\nTest set predictions:")
        print(f"  Size: {len(df_test)}")
        print(f"  Score range: [{test_scores_raw.min():.4f}, {test_scores_raw.max():.4f}]")
        print(f"  Probability range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    
    # Save model and submission
    if save_model_path:
        joblib.dump(best_model, save_model_path)
        if verbose:
            print(f"Saved best model to: {save_model_path}")
    
    if save_submission_path:
        # Create submission file
        id_col = 'Id' if 'Id' in df_test.columns else None
        if id_col is None:
            submission = pd.DataFrame({
                "Id": np.arange(len(df_test)),
                "target": test_predictions
            })
        else:
            submission = pd.DataFrame({
                "Id": df_test[id_col].values,
                "target": test_predictions
            })
        
        submission.to_csv(save_submission_path, index=False)
        if verbose:
            print(f"Saved submission to: {save_submission_path}")
    
    # Create results summary
    results_df = pd.DataFrame(results)
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'results_df': results_df,
        'test_predictions': test_predictions,
        'validation_metrics': validation_metrics,
        'best_prob_method': best_prob_method,
        'preprocessing_pipeline': preprocess
    }

def train_model_by_name(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    param_grid: Optional[Dict[str, Any]] = None,
    preprocessing_pipeline: Optional[Callable] = None,
    scoring_weights: Dict[str, float] = None,
    probability_methods: list = None,
    save_model_path: str = None,
    save_submission_path: str = None,
    verbose: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Convenience function to train a model by name using default parameters.
    
    Args:
        model_name: Name of the model ('IsolationForest', 'OneClassSVM', 'LocalOutlierFactor')
        train_data_path: Path to training data CSV
        val_data_path: Path to validation data CSV  
        test_data_path: Path to test data CSV
        param_grid: Custom parameter grid (if None, uses default for the model)
        preprocessing_pipeline: Function that returns preprocessing pipeline
        scoring_weights: Weights for composite scoring
        probability_methods: List of probability conversion methods to test
        save_model_path: Path to save the best model
        save_submission_path: Path to save submission file
        verbose: Whether to print progress information
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing training results
    """
    
    # Get model class and default parameter grid
    model_class = get_model_class(model_name)
    if param_grid is None:
        param_grid = get_default_param_grid(model_name)
    
    # Set default save paths if not provided
    if save_model_path is None:
        save_model_path = f'results/best_{model_name.lower()}_model.pkl'
    if save_submission_path is None:
        save_submission_path = f'results/{model_name.lower()}_submission.csv'
    
    return train_anomaly_detection_model(
        model_class=model_class,
        param_grid=param_grid,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        preprocessing_pipeline=preprocessing_pipeline,
        scoring_weights=scoring_weights,
        probability_methods=probability_methods,
        save_model_path=save_model_path,
        save_submission_path=save_submission_path,
        verbose=verbose,
        random_state=random_state
    )
if __name__ == "__main__":
    # Example usage of the new training workflow
    
    print("="*80)
    print("ANOMALY DETECTION TRAINING WORKFLOW EXAMPLES")
    print("="*80)
    
    # Data paths
    train_path = '/Users/wanghelin/Documents/course/CS5344/project/dataset/loans_train.csv'
    val_path = '/Users/wanghelin/Documents/course/CS5344/project/dataset/loans_valid.csv'
    test_path = '/Users/wanghelin/Documents/course/CS5344/project/dataset/loans_test.csv'
    
    # Example 1: Train IsolationForest using the convenience function (recommended)
    print("\n" + "="*60)
    print("EXAMPLE 1: Training IsolationForest (Convenience Function)")
    print("="*60)
    
    results_if = train_model_by_name(
        model_name='IsolationForest',
        train_data_path=train_path,
        val_data_path=val_path,
        test_data_path=test_path,
        verbose=True
    )
    
    print(f"\nIsolationForest Results:")
    print(f"Best parameters: {results_if['best_params']}")
    print(f"Best validation score: {results_if['best_score']:.4f}")
    print(f"Best probability method: {results_if['best_prob_method']}")
    
    # Example 2: Train OneClassSVM with custom parameters
    print("\n" + "="*60)
    print("EXAMPLE 2: Training OneClassSVM (Custom Parameters)")
    print("="*60)
    
    # Custom parameter grid for OneClassSVM
    custom_param_grid = {
        "kernel": ['rbf', 'poly'],
        "gamma": ['scale', 0.1],
        "nu": [0.1, 0.15, 0.2]
    }
    
    results_ocsvm = train_model_by_name(
        model_name='OneClassSVM',
        train_data_path=train_path,
        val_data_path=val_path,
        test_data_path=test_path,
        param_grid=custom_param_grid,
        save_model_path='results/best_ocsvm_model.pkl',
        save_submission_path='results/ocsvm_submission.csv',
        verbose=True
    )
    
    print(f"\nOneClassSVM Results:")
    print(f"Best parameters: {results_ocsvm['best_params']}")
    print(f"Best validation score: {results_ocsvm['best_score']:.4f}")
    
    # Example 3: Train LocalOutlierFactor using the full function
    print("\n" + "="*60)
    print("EXAMPLE 3: Training LocalOutlierFactor (Full Function)")
    print("="*60)
    
    results_lof = train_anomaly_detection_model(
        model_class=LocalOutlierFactor,
        param_grid=get_default_param_grid('LocalOutlierFactor'),
        train_data_path=train_path,
        val_data_path=val_path,
        test_data_path=test_path,
        save_model_path='results/best_lof_model.pkl',
        save_submission_path='results/lof_submission.csv',
        verbose=True
    )
    
    print(f"\nLocalOutlierFactor Results:")
    print(f"Best parameters: {results_lof['best_params']}")
    print(f"Best validation score: {results_lof['best_score']:.4f}")
    
    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    models = [
        ('IsolationForest', results_if),
        ('OneClassSVM', results_ocsvm),
        ('LocalOutlierFactor', results_lof)
    ]
    
    print(f"{'Model':<20} {'Best Score':<12} {'ROC-AUC':<8} {'PR-AUC':<8} {'Precision@k':<12}")
    print("-" * 70)
    
    for model_name, results in models:
        metrics = results['validation_metrics']
        print(f"{model_name:<20} {results['best_score']:<12.4f} "
              f"{metrics['roc_auc']:<8.4f} {metrics['pr_auc']:<8.4f} "
              f"{metrics['precision_at_k']:<12.4f}")
    
    # Find best overall model
    best_model_name, best_results = max(models, key=lambda x: x[1]['best_score'])
    print(f"\nBest overall model: {best_model_name} (Score: {best_results['best_score']:.4f})")
    
    print(f"\nAll models trained successfully!")
    print(f"Check the 'results/' directory for saved models and submissions.")