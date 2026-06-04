"""
src/models.py
Factory functions for our 4 classifiers.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

def get_naive_bayes():
    """Fast baseline. Note: Only works with non-negative features (TF-IDF)."""
    return MultinomialNB(alpha=1.0)

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

def get_all_classifiers():
    """Returns a dictionary of all classifiers."""
    return {
        'naive_bayes': get_naive_bayes(),
        'logistic_regression': get_logistic_regression(),
        'linearsvc': get_linearsvc(),
        'xgboost': get_xgboost()
    }