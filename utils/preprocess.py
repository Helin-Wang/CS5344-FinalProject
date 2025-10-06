import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# -----------------------------
# Custom transformers
# -----------------------------

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = list(columns)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        existing_columns = [c for c in self.columns if c in X.columns]
        return X.drop(columns=existing_columns)

class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping, columns):
        self.mapping = dict(mapping)
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].map(self.mapping)
        return X

class OneHotLowCard(BaseEstimator, TransformerMixin):
    """get_dummies on low-card features; stores columns_ so valid set aligns to train."""
    def __init__(self, columns):
        self.columns = list(columns)
        self.columns_ = None  # learned after fit

    def fit(self, X, y=None):
        X_tmp = pd.get_dummies(X, columns=[c for c in self.columns if c in X.columns])
        self.columns_ = list(X_tmp.columns)
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=[c for c in self.columns if c in X.columns])
        # align to training columns; fill missing with 0 and drop extras
        X = X.reindex(columns=self.columns_, fill_value=0)
        return X
    
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Log(1 + frequency) encoding for high-cardinality categoricals."""
    def __init__(self, columns):
        self.columns = list(columns)
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                vc = X[col].value_counts(dropna=False)
                self.freq_maps_[col] = vc.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                fmap = self.freq_maps_.get(col, {})
                # map unseen to 0 -> log1p(0) = 0
                X[col] = X[col].map(lambda x: np.log1p(fmap.get(x, 0)))
        return X

class NumericCleanerImputer(BaseEstimator, TransformerMixin):
    """
    Replaces 9999/999 with NaN, then fills with train mean for specified numeric features.
    """
    def __init__(self, features):
        self.features = list(features)
        self.means_ = None

    def fit(self, X, y=None):
        X_tmp = X.copy()
        existing = [c for c in self.features if c in X_tmp.columns]
        X_tmp[existing] = X_tmp[existing].replace({9999: np.nan, 999: np.nan})
        self.means_ = X_tmp[existing].mean()
        return self

    def transform(self, X):
        X = X.copy()
        existing = [c for c in self.features if c in X.columns]
        X[existing] = X[existing].replace({9999: np.nan, 999: np.nan})
        X[existing] = X[existing].fillna(self.means_)
        return X

class EnhancedDynamicFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Enhanced feature engineering with more sophisticated temporal patterns:
      - Payment behavior analysis (volatility, trends, decline rates)
      - Interest rate stability and changes
      - LTV progression analysis
      - Loan age progression anomalies
      - Financial ratios and risk indicators
      - Temporal pattern analysis
    """
    def __init__(self, months=14):
        self.months = months
        
    def fit(self, X, y=None):
        return self

    def _has_cols(self, X, base):
        return all(f"{i}_{base}" in X.columns for i in range(self.months))

    def transform(self, X):
        X = X.copy()
        m = self.months

        # ----- 1. Payment Behavior Features -----
        upb_cols = [f"{i}_CurrentActualUPB" for i in range(m) if f"{i}_CurrentActualUPB" in X.columns]
        ib_cols = [f"{i}_InterestBearingUPB" for i in range(m) if f"{i}_InterestBearingUPB" in X.columns]
        pca = PCA(n_components=1)
        X["UPB_PCA"] = pca.fit_transform(X[ib_cols+upb_cols])
        X.drop(columns=ib_cols+upb_cols, inplace=True, errors="ignore")
        
        # ----- 2. Interest Rate Analysis -----
        ir_cols = [f"{i}_CurrentInterestRate" for i in range(m) if f"{i}_CurrentInterestRate" in X.columns]
        X.drop(columns=ir_cols, inplace=True, errors="ignore")

        # ----- 3. LTV Progression Analysis -----
        ltv_cols = [f"{i}_EstimatedLTV" for i in range(m) if f"{i}_EstimatedLTV" in X.columns]
        X['LTV_Increase'] = X[ltv_cols[-1]] - X[ltv_cols[0]]
        pca = PCA(n_components=4)
        X[[f"LTV_PCA_{i}" for i in range(4)]] = pca.fit_transform(X[ltv_cols])
        X.drop(columns=ltv_cols, inplace=True, errors="ignore")

        # ----- 4. Loan Age Progression Anomalies -----
        age_cols = [f"{i}_LoanAge" for i in range(m) if f"{i}_LoanAge" in X.columns]
        # Use one new column to specify whether LoanAge increase monotonically
        loan_ages = X[age_cols].to_numpy()
        is_monotonic = np.all(np.diff(loan_ages, axis=1) >= 0, axis=1)
        X['LoanAge_Monotonic'] = is_monotonic
        X.drop(columns=age_cols, inplace=True, errors="ignore")

        # ----- 6. Risk Indicators -----
        X['High_LTV_Risk'] = (X['OriginalLTV'] > 80).astype(int) if 'OriginalLTV' in X.columns else 0
        X['High_DTI_Risk'] = (X['OriginalDTI'] > 40).astype(int) if 'OriginalDTI' in X.columns else 0
  
        # ----- 7. Temporal Pattern Analysis -----
        # if upb_cols:
        #     early_months = min(6, len(upb_cols))
        #     late_months = max(0, len(upb_cols) - 6)
            
        #     if early_months > 0:
        #         early_cols = upb_cols[:early_months]
        #         X['Early_Payment_Issues'] = X[early_cols].std(axis=1)
        #     else:
        #         X['Early_Payment_Issues'] = 0
                
        #     if late_months > 0:
        #         late_cols = upb_cols[-late_months:]
        #         X['Late_Payment_Issues'] = X[late_cols].std(axis=1)
        #     else:
        #         X['Late_Payment_Issues'] = 0
        # else:
        #     X['Early_Payment_Issues'] = 0
        #     X['Late_Payment_Issues'] = 0

        # ----- 8. Max Non-Interest Bearing UPB -----
        cnib_cols = [f"{i}_CurrentNonInterestBearingUPB" for i in range(m)
                     if f"{i}_CurrentNonInterestBearingUPB" in X.columns]
        X['MaxCurrentNonInterestBearingUPB'] = X[cnib_cols].max(axis=1)
        X.drop(columns=cnib_cols, inplace=True, errors="ignore")

        # ----- 9. Drop remaining monthly columns -----
        remaining_monthly = ['MonthlyReportingPeriod', 'RemainingMonthsToLegalMaturity']
        to_drop = [f"{i}_{feat}" for feat in remaining_monthly for i in range(m)]
        X.drop(columns=[c for c in to_drop if c in X.columns], inplace=True, errors="ignore")

        return X

class ScaleSpecificFeatures(BaseEstimator, TransformerMixin):
    """
    Applies StandardScaler only to specified columns.
    """
    def __init__(self, features):
        self.features = list(features)
        self.scaler_ = None

    def fit(self, X, y=None):
        cols = [c for c in self.features if c in X.columns]
        self.scaler_ = StandardScaler().fit(X[cols])
        self.features_ = cols  # keep only those present
        return self

    def transform(self, X):
        X = X.copy()
        if self.features_:
            X[self.features_] = X[self.features_].astype(float)
            X.loc[:, self.features_] = self.scaler_.transform(X[self.features_])
        return X

# -----------------------------
# Build the end-to-end pipeline
# -----------------------------

def make_baseline_preprocessing_pipeline():
    """
    Baseline preprocessing pipeline that only removes irrelevant columns and converts 
    data to values that can be processed by models, without any manual feature engineering.
    """
    # 1) columns to drop - only remove clearly irrelevant columns
    features_to_drop = [
        'FirstPaymentDate',
        'MaturityDate',
        'PPM_Flag',
        'InterestOnlyFlag',
        'ProductType',
        'ReliefRefinanceIndicator',
        'PreHARP_Flag',
        'MSA',
        'PostalCode',
    ]

    # 2) binary columns & mapping - convert Y/N to 1/0
    binary_cols = ['BalloonIndicator', 'FirstTimeHomebuyerFlag', 'SuperConformingFlag']
    binary_mapping = {
        'Y': 1,
        'N': 0,
        '7': -1,     # Unknown
        '99': -1,    # Unknown
        np.nan: -1   # Unknown
    }

    # 3) categoricals - separate low and high cardinality
    low_card_categoricals = [
        'OccupancyStatus', 'Channel', 'PropertyType', 'LoanPurpose', 
        'ProgramIndicator', 'PropertyValMethod'
    ]
    high_card_categoricals = [
        'PropertyState', 'SellerName', 'ServicerName'
    ]

    # 4) numeric features to clean/fill - handle missing values in numeric columns
    numeric_features = [
        'CreditScore', 'MI_Pct', 'OriginalCLTV', 'OriginalDTI', 
        'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 
        'NumberOfBorrowers', 'NumberOfUnits', 'OriginalLoanTerm'
    ]

    pipe = Pipeline(steps=[
        ("drop_columns", DropColumns(features_to_drop)),
        ("binary_map", BinaryMapper(mapping=binary_mapping, columns=binary_cols)),
        ("onehot_low_card", OneHotLowCard(columns=low_card_categoricals)),
        ("freq_encode_high_card", FrequencyEncoder(columns=high_card_categoricals)),
        ("numeric_clean_impute", NumericCleanerImputer(features=numeric_features)),
    ])

    return pipe

def make_preprocessing_pipeline():
    # 1) columns to drop
    features_to_drop = [
        'FirstPaymentDate',
        'MaturityDate',
        'PPM_Flag',
        'InterestOnlyFlag',
        'ProductType',
        'ReliefRefinanceIndicator',
        'PreHARP_Flag',
        'MSA',
        'PostalCode',
    ]

    # 2) binary columns & mapping
    binary_cols = ['BalloonIndicator', 'FirstTimeHomebuyerFlag', 'SuperConformingFlag']
    binary_mapping = {
        'Y': 1,
        'N': 0,
        '7': -1,     # Unknown
        '99': -1,    # Unknown
        np.nan: -1   # Unknown
    }

    # 3) categoricals
    low_card = ['OccupancyStatus', 'Channel', 'PropertyType', 'LoanPurpose', 'ProgramIndicator', 'PropertyValMethod']
    high_card = ['PropertyState', 'SellerName', 'ServicerName', ]

    # 4) numeric features to clean/fill
    static_numeric_features = [
        'CreditScore',
        'MI_Pct',
        'OriginalCLTV',
        'OriginalDTI',
        'OriginalUPB',
        'OriginalLTV',
        'OriginalInterestRate',
        'NumberOfBorrowers',
        'NumberOfUnits',
        'OriginalLoanTerm'
    ]
    
    # 5) features to scale
    features_to_scale = static_numeric_features + [
        'LTV_Increase',
        'LTV_PCA_1',
        'LTV_PCA_2',
        'LTV_PCA_3',
        'LTV_PCA_4',
        
        'UPB_PCA',
        'Final_CurrentActualUPB',
        'Final_InterestBearingUPB',
        
        'Interest_to_Principal',
        
        'MaxCurrentNonInterestBearingUPB',
    ]

    pipe = Pipeline(steps=[
        ("drop_columns", DropColumns(features_to_drop)),
        ("binary_map", BinaryMapper(mapping=binary_mapping, columns=binary_cols)),
        ("onehot_low_card", OneHotLowCard(columns=low_card)),
        ("freq_encode_high_card", FrequencyEncoder(columns=high_card)),
        ("numeric_clean_impute", NumericCleanerImputer(features=static_numeric_features)),
        ("dynamic_features", EnhancedDynamicFeaturesEngineer(months=14)),
        ("scale_selected_numeric", ScaleSpecificFeatures(features=features_to_scale)),
    ])

    return pipe

def split_dataset(df):
    X = df.drop(columns=["target", "index"])
    y = df["target"]
    
    return X, y
