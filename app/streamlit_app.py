"""
app/streamlit_app.py
Interactive Streamlit Demo for IMDb Sentiment Classification
"""
import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IMDb Sentiment Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=150)
    st.title("🎬 IMDb Sentiment")
    st.markdown("---")
    st.markdown("**Model:** TF-IDF + LinearSVC")
    st.markdown("**Features:** Unigrams + Bigrams")
    st.markdown("**Accuracy:** ~90%")
    st.markdown("---")
    st.info("💡 **Try these examples:**\n\n- 'This movie was absolutely brilliant!'\n- 'Terrible waste of time'\n- 'Not bad, but not great either'")

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🎬 IMDb Movie Review Sentiment Classifier</h1>', unsafe_allow_html=True)
st.markdown("Enter a movie review below to predict whether it's **positive** or **negative**.")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "Movie Review:",
        height=150,
        placeholder="Type or paste a movie review here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Sentiment", type="primary", use_container_width=True)

# Example reviews
st.markdown("---")
st.markdown("**📝 Try an example:**")
examples = {
    "Positive": "This movie was absolutely brilliant! The acting was superb, the cinematography was stunning, and the storyline kept me hooked from beginning to end. Highly recommended!",
    "Negative": "Complete waste of time. Terrible acting, predictable plot, and awful dialogue. I couldn't wait for it to end. Avoid at all costs.",
    "Mixed": "It wasn't the worst movie I've seen, but it certainly wasn't the best. Some good moments, but overall forgettable. The lead actor tried hard, but the script let them down."
}

example_choice = st.selectbox("Select an example:", list(examples.keys()))
if st.button("Load Example"):
    user_input = examples[example_choice]
    st.rerun()

# Prediction
if predict_btn and user_input.strip():
    with st.spinner("🤖 Analyzing sentiment..."):
        try:
            # Call FastAPI
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": user_input},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_color = "🟢" if result['sentiment'] == 'positive' else "🔴"
                    st.metric("Sentiment", f"{sentiment_color} {result['sentiment'].title()}")
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                with col3:
                    st.metric("Inference Time", f"{result['inference_time_ms']:.2f} ms")
                
                # Confidence bar
                st.markdown("### 📈 Confidence Distribution")
                probs = result['probabilities']
                fig, ax = plt.subplots(figsize=(8, 2))
                colors = ['#dc3545' if probs['negative'] > probs['positive'] else '#28a745',
                         '#28a745' if probs['positive'] > probs['negative'] else '#dc3545']
                ax.barh(['Negative', 'Positive'], [probs['negative'], probs['positive']], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Class Probabilities')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed info
                with st.expander("🔍 Detailed Information"):
                    st.json(result)
                
                # Visual feedback
                if result['sentiment'] == 'positive':
                    st.success(f"✅ This review is **{result['sentiment'].upper()}** with {result['confidence']*100:.1f}% confidence!")
                else:
                    st.error(f"❌ This review is **{result['sentiment'].upper()}** with {result['confidence']*100:.1f}% confidence!")
                    
            else:
                st.error(f"API Error: {response.status_code}")
                st.error(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
            st.info("💡 Start the API with: `python -m app.api`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif predict_btn and not user_input.strip():
    st.warning("⚠️ Please enter a review first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p>Built with FastAPI + Streamlit | IMDb Sentiment Analysis Project</p>
</div>
""", unsafe_allow_html=True)