# 🎬 IMDb Sentiment Analysis - MLOps Project

A production-grade sentiment classification system built with modern MLOps practices.

## 📋 Project Overview

This project implements a complete machine learning pipeline for classifying IMDb movie reviews as positive or negative. It follows industry-standard MLOps practices including modular architecture, comprehensive testing, and deployment-ready code.

## ✨ Features

- **Advanced Text Preprocessing**: POS-aware lemmatization, contraction handling, and elongated word normalization
- **Multiple Vectorization Strategies**: TF-IDF (unigrams/bigrams), Word2Vec, and Sentence-BERT
- **12 Model Variants**: 3 vectorizers × 4 classifiers (Naive Bayes, Logistic Regression, LinearSVC, XGBoost)
- **Comprehensive Evaluation**: Accuracy, F1-Score, ROC-AUC, Confusion Matrix
- **Error Analysis**: Automated error categorization and taxonomy
- **Explainability**: LIME and SHAP integration for model interpretability
- **Production API**: FastAPI REST API with Streamlit demo interface

## 🏗️ Architecture
├── src/ # Core source code
│ ├── data_pipeline.py # Phase 1: Data ingestion & preprocessing
│ ├── vectorizers.py # Phase 2: Feature engineering
│ ├── models.py # Phase 2: Classifier definitions
│ ├── train.py # Phase 2: Training pipeline
│ ├── evaluate.py # Phase 3: Model evaluation
│ └── explainability.py # Phase 3: LIME/SHAP analysis
├── app/ # Deployment
│ ├── api.py # FastAPI REST API
│ └── streamlit_app.py # Interactive demo
├── notebooks/ # EDA and experimentation
├── tests/ # Unit tests
└── outputs/ # Generated reports & visualizations


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"

Usage
1. Run the Data Pipeline (Phase 1)
bash
1
2. Train Models (Phase 2)
bash
1
3. Evaluate Models (Phase 3)
bash
1
4. Generate Explainability Reports
bash
1
5. Launch the API
bash
1
6. Launch the Streamlit Demo
bash
1
📊 Model Performance
Model
Vectorizer
Test Accuracy
Test F1
Inference Time
LinearSVC
TF-IDF (Uni+Bi)
~90.4%
~90.4%
0.26 ms
Logistic Regression
TF-IDF (Uni+Bi)
~90.2%
~90.2%
0.30 ms
XGBoost
Word2Vec
~87.5%
~87.4%
1.2 ms
🧪 Testing
Run the test suite:
bash
1
📁 Project Structure
src/: Core machine learning pipeline
data_pipeline.py: Advanced text cleaning with POS-aware lemmatization
vectorizers.py: TF-IDF, Word2Vec, SBERT implementations
models.py: Classifier factory functions
train.py: Automated training of 12 model variants
evaluate.py: Comprehensive evaluation on locked test set
explainability.py: LIME/SHAP explanations
app/: Production deployment
api.py: FastAPI REST API with validation
streamlit_app.py: Interactive web demo
notebooks/: Exploratory data analysis
tests/: Unit tests for data pipeline
outputs/: Generated visualizations and reports
🔑 Key Highlights
✅ Google-Level MLOps: Modular, testable, production-ready code
✅ Error Taxonomy: Automated categorization of model failures
✅ No Data Leakage: Strict train/val/test separation with sklearn Pipelines
✅ Explainable AI: LIME and SHAP integration for transparency
✅ Production API: FastAPI with Pydantic validation and CORS
✅ Comprehensive Testing: Unit tests for critical components
📄 License
MIT License
👤 Author
Jay Patil - Data Science Intern @ NeoSoft Technologies
📧 Contact
jay71patil@gmail.com