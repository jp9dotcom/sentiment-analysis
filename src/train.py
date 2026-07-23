"""
src/train.py
Master training script for Phase 2: TF-IDF, Word2Vec, and SBERT.
"""
import os
import time
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.models import get_all_classifiers
from src.vectorizers import get_tfidf, get_word2vec, get_sbert

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("Loading Phase 1 data...")
    X_train = np.load(f'{DATA_DIR}/X_train.npy', allow_pickle=True)
    X_val = np.load(f'{DATA_DIR}/X_val.npy', allow_pickle=True)
    y_train = np.load(f'{DATA_DIR}/y_train.npy')
    y_val = np.load(f'{DATA_DIR}/y_val.npy')
    return X_train, X_val, y_train, y_val

def train_and_save(pipeline, X_train, y_train, X_val, y_val, pipeline_name):
    """Helper function to train, evaluate, and save a pipeline."""
    print(f"  ⏳ Training: {pipeline_name} ...")
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"     ✅ {pipeline_name:<30} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.1f}s")
    
    model_path = f"{MODEL_DIR}/{pipeline_name}.pkl"
    joblib.dump(pipeline, model_path)
    
    return {
        'model': pipeline_name,
        'accuracy': acc,
        'f1_score': f1,
        'train_time': train_time
    }

def run_training():
    from sklearn.base import clone
    X_train, X_val, y_train, y_val = load_data()
    classifiers = get_all_classifiers()
    results = []
    
    print(f"\n{'='*60}")
    print("PHASE 2: TRAINING ALL PIPELINES")
    print(f"{'='*60}\n")
    
    # ==========================================
    # 1. TF-IDF (3 N-gram configurations)
    # ==========================================
    ngram_configs = {
        'tfidf_uni': (1, 1),
        'tfidf_bi': (2, 2),
        'tfidf_unibi': (1, 2)
    }
    
    for vec_name, ng_range in ngram_configs.items():
        print(f"🚀 Vectorizer: {vec_name}")
        vectorizer = get_tfidf(ngram_range=ng_range)
        for clf_name, classifier in classifiers.items():
            cloned_clf = clone(classifier)
            pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', cloned_clf)])
            res = train_and_save(pipeline, X_train, y_train, X_val, y_val, f"{vec_name}_{clf_name}")
            results.append(res)

    # ==========================================
    # 2. Word2Vec (Dense Embeddings)
    # ==========================================
    print(f"\n🚀 Vectorizer: word2vec (Dense Embeddings)")
    vec_w2v = get_word2vec()
    for clf_name, classifier in classifiers.items():
        # Skip Naive Bayes for dense embeddings (requires non-negative values)
        if clf_name == 'naive_bayes':
            continue 
        cloned_clf = clone(classifier)
        pipeline = Pipeline([('vectorizer', vec_w2v), ('classifier', cloned_clf)])
        res = train_and_save(pipeline, X_train, y_train, X_val, y_val, f"word2vec_{clf_name}")
        results.append(res)

    # ==========================================
    # 3. Sentence-BERT (Contextual Embeddings)
    # ==========================================
    print(f"\n Vectorizer: sbert (Contextual Embeddings)")
    vec_sbert = get_sbert()
    for clf_name, classifier in classifiers.items():
        # Skip Naive Bayes for dense embeddings
        if clf_name == 'naive_bayes':
            continue 
        cloned_clf = clone(classifier)
        pipeline = Pipeline([('vectorizer', vec_sbert), ('classifier', cloned_clf)])
        res = train_and_save(pipeline, X_train, y_train, X_val, y_val, f"sbert_{clf_name}")
        results.append(res)

    # Print Master Leaderboard
    df_results = pd.DataFrame(results).sort_values(by='f1_score', ascending=False)
    print("\n=== PHASE 2 MASTER LEADERBOARD (Validation Set) ===")
    print(df_results.to_string(index=False))
    df_results.to_csv(f"{MODEL_DIR}/leaderboard.csv", index=False)
    print(f"\n✓ Saved leaderboard to {MODEL_DIR}/leaderboard.csv")

    # Select and save best model based on validation F1 score
    best_row = df_results.iloc[0]
    best_model_name = best_row['model']
    best_f1 = best_row['f1_score']
    best_acc = best_row['accuracy']
    best_train_time = best_row['train_time']
    
    best_model_path = f"{MODEL_DIR}/{best_model_name}.pkl"
    best_pipeline = joblib.load(best_model_path)
    
    best_model_save_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_pipeline, best_model_save_path)
    
    # Save best model metrics
    best_metrics = {
        'model': best_model_name,
        'accuracy': float(best_acc),
        'f1_score': float(best_f1),
        'train_time': float(best_train_time)
    }
    best_metrics_path = f"{MODEL_DIR}/best_model_metrics.json"
    with open(best_metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print(f"   Validation F1: {best_f1:.4f} | Accuracy: {best_acc:.4f}")
    print(f"   Saved to: {best_model_save_path}")
    print(f"   Metrics saved to: {best_metrics_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_training()