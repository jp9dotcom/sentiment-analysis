"""
app/api.py
Production FastAPI REST API for IMDb Sentiment Classification
"""
import os
import re
import time
import joblib
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# NLP Setup (Must match Phase 1 exactly!)
# ─────────────────────────────────────────────────────────────────────────────

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
NEGATIONS = {'not', 'no', 'nor', 'never', "n't", 'don', 'doesn', 'didn', 'won', 'wouldn', 'shouldn'}
STOP_WORDS = STOP_WORDS - NEGATIONS
LEMMATIZER = WordNetLemmatizer()

CONTRACTION_MAP = {
    "n't": " not", "'re": " are", "'s": " is", "'d": " would", 
    "'ll": " will", "'ve": " have", "'m": " am"
}

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return 'a'
    elif treebank_tag.startswith('V'): return 'v'
    elif treebank_tag.startswith('N'): return 'n'
    elif treebank_tag.startswith('R'): return 'r'
    else: return 'n'

def clean_text(text: str) -> str:
    """Production-grade text cleaning (identical to Phase 1)."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    for k, v in CONTRACTION_MAP.items():
        text = text.replace(k, v)
        
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    import nltk
    pos_tags = nltk.pos_tag(tokens)
    
    cleaned_tokens = []
    for word, tag in pos_tags:
        if len(word) > 2 and word not in STOP_WORDS:
            wn_pos = get_wordnet_pos(tag)
            lemma = LEMMATIZER.lemmatize(word, pos=wn_pos)
            cleaned_tokens.append(lemma)
            
    return ' '.join(cleaned_tokens)

# ─────────────────────────────────────────────────────────────────────────────
# Load Best Model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_unibi_logistic_regression.pkl')  # Update to your best model

if not os.path.exists(MODEL_PATH):
    # Auto-find the best model
    for f in os.listdir(MODEL_DIR):
        if f.endswith('regression.pkl'):
            MODEL_PATH = os.path.join(MODEL_DIR, f)
            break

print(f"🚀 Loading model: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Production-grade sentiment classification API",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, 
                      description="Movie review text to analyze")

class PredictionResponse(BaseModel):
    review_snippet: str
    sentiment: str  # "positive" or "negative"
    confidence: float
    probabilities: dict
    inference_time_ms: float
    model_used: str

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model": os.path.basename(MODEL_PATH),
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment of a movie review.
    """
    try:
        # 1. Clean the text (MUST match training preprocessing)
        cleaned_text = clean_text(request.text)
        
        # 2. Predict
        t0 = time.perf_counter()
        pred_label = int(pipeline.predict([cleaned_text])[0])
        proba = pipeline.predict_proba([cleaned_text])[0]
        inference_time = (time.perf_counter() - t0) * 1000
        
        # 3. Format response
        sentiment = "positive" if pred_label == 1 else "negative"
        confidence = float(max(proba))
        
        return {
            "review_snippet": request.text[:100] + ("..." if len(request.text) > 100 else ""),
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                "negative": round(float(proba[0]), 4),
                "positive": round(float(proba[1]), 4)
            },
            "inference_time_ms": round(inference_time, 3),
            "model_used": os.path.basename(MODEL_PATH)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
def model_info():
    """Get information about the loaded model."""
    return {
        "model_path": MODEL_PATH,
        "preprocessing": "Advanced NLP pipeline with POS-aware lemmatization",
        "features": "TF-IDF unigrams+bigrams (max 20k features)",
        "classifier": "LinearSVC with isotonic calibration"
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)