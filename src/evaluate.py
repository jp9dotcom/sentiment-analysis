"""
src/evaluate.py
Phase 3: Evaluates ALL models on the locked Test set, builds the final comparison table,
and runs deep error analysis on the absolute best model.
"""
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Error Categorization Logic
# ─────────────────────────────────────────────────────────────────────────────
def categorize_error(review_text):
    text = str(review_text).lower()
    words = text.split()
    
    if len(words) < 20: return "Short Review (Lack of Context)"
    if re.search(r'\b(not|never|no|neither|without|didn\'t|wasn\'t|isn\'t)\b', text):
        return "Negation / Complex Grammar"
    
    positive_cues = ['good', 'great', 'best', 'amazing', 'excellent', 'love']
    negative_cues = ['bad', 'worst', 'terrible', 'awful', 'hate', 'boring']
    if any(w in text for w in positive_cues) and any(w in text for w in negative_cues):
        return "Mixed Sentiment / Sarcasm"
        
    return "General Confusion"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Evaluation Pipeline
# ────────────────────────────────────────────────────────────────────────────
def run_evaluation():
    print("="*60)
    print("PHASE 3: FINAL EVALUATION ON LOCKED TEST SET")
    print("="*60)
    
    # Load locked test data
    X_test = np.load(f'{DATA_DIR}/X_test.npy', allow_pickle=True)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    
    # Load the Validation Leaderboard from Phase 2
    leaderboard_path = f'{MODEL_DIR}/leaderboard.csv'
    if not os.path.exists(leaderboard_path):
        raise FileNotFoundError("Run train.py first to generate the leaderboard!")
        
    val_leaderboard = pd.read_csv(leaderboard_path)
    
    final_results = []
    
    # ==========================================
    # STEP A: Test ALL models on the Test Set
    # ==========================================
    print(f"\n🧪 Testing {len(val_leaderboard)} models on the locked Test Set...")
    for index, row in val_leaderboard.iterrows():
        model_name = row['model']
        model_path = f'{MODEL_DIR}/{model_name}.pkl'
        
        if not os.path.exists(model_path):
            continue
            
        print(f"    Testing {model_name}...")
        pipeline = joblib.load(model_path)
        
        # Measure Inference Speed
        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        inference_time = time.perf_counter() - t0
        ms_per_sample = (inference_time / len(X_test)) * 1000
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        final_results.append({
            'model': model_name,
            'test_accuracy': round(acc, 4),
            'test_f1': round(f1, 4),
            'inference_ms': round(ms_per_sample, 3),
            'val_f1': row['f1_score'] # Keep validation score for comparison
        })
        print(f"      ✅ Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Speed: {ms_per_sample:.3f} ms/sample")

    # Save Final Test Leaderboard
    df_final = pd.DataFrame(final_results).sort_values(by='test_f1', ascending=False)
    final_csv_path = os.path.join(OUTPUT_DIR, 'final_test_leaderboard.csv')
    df_final.to_csv(final_csv_path, index=False)
    print(f"\n✅ Saved final test leaderboard to {final_csv_path}")
    
    # ==========================================
    # STEP B: Deep Error Analysis on the BEST Model
    # ==========================================
    best_model_name = df_final.iloc[0]['model']
    print(f"\n🏆 Absolute Best Model on Test Set: {best_model_name}")
    print(f"   Running deep error taxonomy and generating reports...")
    
    best_model_path = f'{MODEL_DIR}/{best_model_name}.pkl'
    pipeline = joblib.load(best_model_path)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{best_model_name}.png'), dpi=150)
    plt.close()
    
    # 2. Error Taxonomy Dataset
    error_indices = np.where(y_pred != y_test)[0]
    errors = []
    for idx in error_indices:
        true_label = "positive" if y_test[idx] == 1 else "negative"
        pred_label = "positive" if y_pred[idx] == 1 else "negative"
        confidence = float(max(y_prob[idx]))
        category = categorize_error(X_test[idx])
        if confidence > 0.85: category = f"Highly Confident Error ({category})"
            
        errors.append({
            'review_text': X_test[idx],
            'true_label': true_label, 'predicted_label': pred_label,
            'confidence_score': round(confidence, 4), 'error_category': category
        })
        
    df_errors = pd.DataFrame(errors)
    df_errors.to_csv(os.path.join(OUTPUT_DIR, 'misclassified_reviews.csv'), index=False)
    
    # 3. Plot Error Categories
    plt.figure(figsize=(10, 5))
    counts = df_errors['error_category'].value_counts()
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title('Error Taxonomy: Why the Model Failed', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_taxonomy.png'), dpi=150)
    plt.close()
    
    print(f"✅ Saved Confusion Matrix, Misclassified Dataset, and Error Taxonomy to {OUTPUT_DIR}/")
    print("="*60)
    print("✅ PHASE 3 COMPLETE! Next: Run src/explainability.py for LIME.")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()