import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(
    model, 
    model_name: str, 
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    feature_names: list = None,
    top_n: int = 20,
    save_plot: bool = True,
    plot_path: str = None
) -> Dict[str, Any]:
    """
    Analyze feature importance for different anomaly detection models.
    
    Args:
        model: Trained model
        model_name: Name of the model ('IsolationForest', 'OneClassSVM', 'LocalOutlierFactor')
        X_train: Training features
        X_val: Validation features  
        y_val: Validation labels
        feature_names: List of feature names (if None, uses column names)
        top_n: Number of top features to return
        save_plot: Whether to save importance plot
        plot_path: Path to save plot (if None, auto-generates)
        
    Returns:
        Dictionary containing feature importance results
    """
    
    if feature_names is None:
        feature_names = list(X_train.columns)
    
    importance_results = {
        'model_name': model_name,
        'feature_names': feature_names,
        'importance_scores': None,
        'top_features': None,
        'importance_method': None
    }
    
    try:
        if model_name == 'IsolationForest':
            # IsolationForest has built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                importance_method = 'built_in'
            else:
                # Fallback to permutation importance
                perm_importance = permutation_importance(
                    model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1,
                    scoring='average_precision'
                )
                importance_scores = perm_importance.importances_mean
                importance_method = 'permutation'
                
        elif model_name == 'OneClassSVM':
            # OneClassSVM doesn't have built-in importance, use permutation
            perm_importance = permutation_importance(
                model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1,
                scoring='average_precision'
            )
            importance_scores = perm_importance.importances_mean
            importance_method = 'permutation'
            
        elif model_name == 'LocalOutlierFactor':
            # LOF doesn't have built-in importance, use permutation
            perm_importance = permutation_importance(
                model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1,
                scoring='average_precision'
            )
            importance_scores = perm_importance.importances_mean
            importance_method = 'permutation'
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        importance_results.update({
            'importance_scores': importance_scores,
            'top_features': top_features,
            'importance_method': importance_method,
            'importance_df': importance_df
        })
        
        # Create and save plot if requested
        if save_plot:
            if plot_path is None:
                plot_path = f'results/{model_name.lower()}_feature_importance.png'
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'{model_name} - Top {top_n} Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            importance_results['plot_path'] = plot_path
        
        return importance_results
        
    except Exception as e:
        print(f"Error analyzing feature importance for {model_name}: {e}")
        return importance_results