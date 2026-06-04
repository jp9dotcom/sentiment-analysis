"""
src/explainability.py
Phase 3: Uses LIME to explain exactly WHY the model failed on specific error categories.
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# Configuration
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'

def run_explainability():
    print("="*60)
    print("PHASE 3: LIME EXPLAINABILITY ON MISCLASSIFIED REVIEWS")
    print("="*60)
    
    # 1. Load Data
    error_csv_path = os.path.join(OUTPUT_DIR, 'misclassified_reviews.csv')
    if not os.path.exists(error_csv_path):
        raise FileNotFoundError("Run evaluate.py first to generate misclassified_reviews.csv")
    
    df_errors = pd.read_csv(error_csv_path)
    
    # Load the best model
    leaderboard = pd.read_csv(f'{MODEL_DIR}/leaderboard.csv')
    best_model_name = leaderboard.iloc[0]['model']
    pipeline = joblib.load(f'{MODEL_DIR}/{best_model_name}.pkl')
    
    # 2. Initialize LIME
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    
    # 3. Pick the #1 Most Confident Error from each Category
    categories = df_errors['error_category'].unique()
    
    for category in categories:
        print(f"\n🔍 Analyzing Category: {category}")
        
        # Get the single review with the highest confidence score in this category
        df_cat = df_errors[df_errors['error_category'] == category]
        worst_error = df_cat.loc[df_cat['confidence_score'].idxmax()]
        
        review_text = worst_error['review_text']
        true_label = worst_error['true_label']
        pred_label = worst_error['predicted_label']
        
        print(f"   True: {true_label} | Predicted: {pred_label} (Conf: {worst_error['confidence_score']})")
        print(f"   Review: {review_text[:100]}...")
        
        # 4. Run LIME
        exp = explainer.explain_instance(
            review_text, 
            pipeline.predict_proba, 
            num_features=8, # Show top 8 words that influenced the decision
            num_samples=500
        )
        
        # 5. Save the LIME HTML visualization
        safe_category = category.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        html_path = os.path.join(OUTPUT_DIR, f'lime_explanation_{safe_category}.html')
        exp.save_to_file(html_path)
        print(f"   ✅ Saved LIME HTML report to: {html_path}")

    print("\n" + "="*60)
    print("✅ PHASE 3 EXPLAINABILITY COMPLETE!")
    print("Open the HTML files in your browser to see the highlighted word weights.")
    print("="*60)

if __name__ == "__main__":
    run_explainability()