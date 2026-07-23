"""
src/models.py
Factory functions for our 4 classifiers.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

def get_naive_bayes():
    """Fast baseline. Note: Only works with non-negative features (TF-IDF)."""
    return MultinomialNB(alpha=1.0)

def get_linear_regression():
    """Linear baseline for binary classification using LogisticRegression with default parameters."""
    return LogisticRegression(random_state=42)

def get_logistic_regression():
    """Strong linear baseline."""
    return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

def get_linearsvc():
    """
    Best on sparse TF-IDF. 
    Wrapped in CalibratedClassifierCV because raw LinearSVC lacks predict_proba(), 
    which we need for ROC-AUC and confidence scores later.
    """
    base_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    return CalibratedClassifierCV(base_svc, cv=3, method='isotonic')

def get_xgboost():
    """Ensemble power. tree_method='hist' prevents Out-Of-Memory errors on large sparse matrices."""
    return XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        tree_method='hist',  # Crucial for memory efficiency
        eval_metric='logloss',
        random_state=42
    )

def get_gradient_boosting():
    """Gradient Boosting classifier with calibrated probability estimates."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

def get_random_forest():
    """Random Forest classifier with parallel processing."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

def get_all_classifiers():
    """Returns a dictionary of all classifiers."""
    return {
        'naive_bayes': get_naive_bayes(),
        'linear_regression': get_linear_regression(),
        'logistic_regression': get_logistic_regression(),
        'linearsvc': get_linearsvc(),
        'xgboost': get_xgboost(),
        'gradient_boosting': get_gradient_boosting(),
        'random_forest': get_random_forest()
    }