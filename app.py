import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .fake-news {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Load or create dataset
@st.cache_data
def load_data():
    """Load sample dataset"""
    real_news = [
        'Scientists at Stanford University published research on climate change in Nature journal',
        'The Federal Reserve announced interest rate changes following economic indicators',
        'New study shows correlation between exercise and mental health published in medical journal',
        'President announces new infrastructure bill after congressional approval',
        'University research team receives federal grant for renewable energy project',
        'Economic report shows GDP growth in third quarter according to government data',
        'Medical journal publishes peer-reviewed findings on vaccine efficacy',
        'Climate data shows temperature trends over decades from NASA research',
        'Stock market closes higher after positive earnings reports from major companies',
        'Supreme Court announces ruling on constitutional matter after months of deliberation',
        'International trade agreement signed between multiple nations at summit',
        'Research institute reports breakthrough in cancer treatment trials',
        'Central bank maintains interest rates as inflation remains stable',
        'University study examines effects of sleep on cognitive performance',
        'Government agency releases annual report on economic indicators',
        'Scientific team discovers new species in Amazon rainforest expedition',
        'Health officials recommend updated vaccination guidelines based on research',
        'Technology company announces quarterly earnings meeting analyst expectations',
        'Archaeological discovery sheds light on ancient civilization',
        'Meteorological service issues weather forecast based on satellite data',
    ] * 5
    
    fake_news = [
        'BREAKING NEWS: Shocking conspiracy revealed! Government hiding the truth! Share now!',
        'You won\'t believe what celebrities are doing! Click here for shocking photos!!!',
        'ALERT: Miracle cure discovered! Doctors hate this one simple trick!',
        'EXPOSED: Secret society controlling world events! The truth they don\'t want you to know!',
        'WARNING: This common food is killing you! Share to save lives!!!',
        'SHOCKING truth about vaccines! What doctors aren\'t telling you!',
        'Celebrities using this one weird trick! Must see before banned!',
        'Government hiding aliens at secret base! Leaked documents prove it!',
        'BREAKING: Billionaire admits shocking secret! You won\'t believe what happens next!',
        'URGENT: New world order taking over! Share before deleted!',
        'This mom discovered amazing trick! Doctors are furious!',
        'ALERT: Moon landing was fake! New evidence proves conspiracy!',
        'Miracle diet melts fat overnight! Doctors shocked by results!',
        'EXPOSED: Chemtrails poisoning population! Government cover-up revealed!',
        'Celebrity death hoax! The truth will shock you!',
        'Ancient aliens built pyramids! Scientists confirm discovery!',
        'BREAKING: Flat earth proven! NASA admits lying!',
        'Miracle supplement cures all diseases! Big pharma hiding it!',
        'Government mind control through 5G! Shocking truth exposed!',
        'URGENT: Economic collapse coming! Prepare now!',
    ] * 5
    
    df = pd.DataFrame({
        'text': real_news + fake_news,
        'label': [0]*len(real_news) + [1]*len(fake_news)
    })
    
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

# Train model
@st.cache_resource
def train_model():
    """Train the fake news detection model"""
    df = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy, X_test, y_test, y_pred

# Predict function
def predict_news(text, model, vectorizer):
    """Predict if news is fake or real"""
    cleaned = preprocess_text(text)
    if not cleaned:
        return None
    
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': max(probability) * 100,
        'fake_prob': probability[1] * 100,
        'real_prob': probability[0] * 100
    }

# Initialize model
model, vectorizer, accuracy, X_test, y_test, y_pred = train_model()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=100)
    st.title("📰 Fake News Detector")
    st.markdown("---")
    
    st.subheader("📊 Model Performance")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.info("""
    This AI-powered system uses **Machine Learning** to detect fake news articles.
    
    **Features:**
    - Natural Language Processing
    - TF-IDF Vectorization
    - Logistic Regression Model
    - Real-time Analysis
    """)
    
    st.markdown("---")
    st.subheader("🔍 How It Works")
    st.markdown("""
    1. **Text Preprocessing**: Cleans the input
    2. **Feature Extraction**: Converts to numbers
    3. **ML Classification**: Predicts authenticity
    4. **Confidence Score**: Shows certainty
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📊 Model Stats", "📈 Visualizations", "ℹ️ Info"])

# TAB 1: DETECTOR
with tab1:
    st.title("🔍 Fake News Detection System")
    st.markdown("### Enter a news article to verify its authenticity")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        news_text = st.text_area(
            "📝 Paste your news article here:",
            height=200,
            placeholder="Enter the news headline or full article text...",
            help="Copy and paste any news article or headline you want to verify"
        )
    
    with col2:
        st.markdown("### Quick Examples")
        if st.button("📗 Real News Example"):
            news_text = "Scientists at MIT have developed a new solar panel technology that increases efficiency by 20 percent. The research, published in Nature Energy, shows promising results for renewable energy applications."
            st.rerun()
        
        if st.button("📕 Fake News Example"):
            news_text = "SHOCKING: Government admits aliens exist! You won't believe what they've been hiding! Click here for the truth they don't want you to know!!!"
            st.rerun()
        
        if st.button("🔄 Clear"):
            news_text = ""
            st.rerun()
    
    # Analyze button
    if st.button("🚀 ANALYZE NEWS", type="primary"):
        if not news_text or not news_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("🔍 Analyzing article..."):
                result = predict_news(news_text, model, vectorizer)
                
                if result is None:
                    st.error("❌ Could not analyze the text. Please try different content.")
                else:
                    st.markdown("---")
                    st.markdown("## 📋 Analysis Results")
                    
                    # Results display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['prediction'] == 'FAKE':
                            st.markdown('<div class="fake-news">', unsafe_allow_html=True)
                            st.error("### 🚫 FAKE NEWS")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="real-news">', unsafe_allow_html=True)
                            st.success("### ✅ REAL NEWS")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric(
                            "Confidence Level",
                            f"{result['confidence']:.1f}%",
                            help="How confident the model is in its prediction"
                        )
                    
                    with col3:
                        if result['confidence'] > 80:
                            st.success("🎯 High Confidence")
                        elif result['confidence'] > 60:
                            st.warning("⚠️ Moderate Confidence")
                        else:
                            st.info("❓ Low Confidence")
                    
                    # Detailed probabilities
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Probability Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Real News Probability")
                        st.progress(result['real_prob'] / 100)
                        st.markdown(f"**{result['real_prob']:.2f}%**")
                    
                    with col2:
                        st.markdown("#### 🚫 Fake News Probability")
                        st.progress(result['fake_prob'] / 100)
                        st.markdown(f"**{result['fake_prob']:.2f}%**")
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['real_prob'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Authenticity Score", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "green"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#ffcdd2'},
                                {'range': [50, 75], 'color': '#fff9c4'},
                                {'range': [75, 100], 'color': '#c8e6c9'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Interpretation")
                    
                    if result['prediction'] == 'FAKE':
                        if result['confidence'] > 80:
                            st.error("⚠️ **High likelihood of misinformation.** This article shows strong characteristics of fake news.")
                        else:
                            st.warning("⚠️ **Possible misinformation.** Verify this information with trusted sources.")
                    else:
                        if result['confidence'] > 80:
                            st.success("✅ **Appears authentic.** This article shows characteristics of legitimate news.")
                        else:
                            st.info("ℹ️ **Likely authentic, but verify.** Cross-check with multiple reliable sources.")
                    
                    st.info("📌 **Remember:** No AI is 100% accurate. Always verify important news from multiple reliable sources!")

# TAB 2: MODEL STATS
with tab2:
    st.title("📊 Model Performance Statistics")
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col2:
        precision = (confusion_matrix(y_test, y_pred)[1][1] / 
                    (confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[0][1]))
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        recall = (confusion_matrix(y_test, y_pred)[1][1] / 
                 (confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[1][0]))
        st.metric("Recall", f"{recall*100:.2f}%")
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall)
        st.metric("F1-Score", f"{f1*100:.2f}%")
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Real News', 'Fake News'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

# TAB 3: VISUALIZATIONS
with tab3:
    st.title("📈 Model Visualizations")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Real', 'Fake'],
                    y=['Real', 'Fake'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    
    fig.update_layout(title="Confusion Matrix", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics bar chart
    metrics_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy*100, precision*100, recall*100, f1*100]
    })
    
    fig2 = px.bar(metrics_data, x='Metric', y='Score',
                  title="Model Performance Metrics",
                  color='Score',
                  color_continuous_scale='Greens')
    fig2.update_layout(yaxis_range=[0, 100], height=400)
    st.plotly_chart(fig2, use_container_width=True)

# TAB 4: INFO
with tab4:
    st.title("ℹ️ Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **Machine Learning-based Fake News Detection System** that analyzes 
        news articles and determines their authenticity.
        
        ### 🛠️ Technologies Used
        
        - **Python** - Programming language
        - **Scikit-learn** - Machine learning library
        - **Streamlit** - Web framework
        - **Plotly** - Data visualization
        - **Pandas** - Data manipulation
        - **TF-IDF** - Text vectorization
        
        ### 🧠 How It Works
        
        1. **Text Preprocessing**: Removes URLs, punctuation, special characters
        2. **Vectorization**: Converts text to numerical features using TF-IDF
        3. **Classification**: Uses Logistic Regression to predict
        4. **Probability**: Returns confidence scores
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Limitations
        
        - Model trained on limited dataset
        - May not catch sophisticated fake news
        - Context and source credibility not considered
        - Requires continuous updates
        
        ### 💡 Best Practices
        
        ✅ Use multiple sources for verification  
        ✅ Check publication date and author  
        ✅ Look for reliable news organizations  
        ✅ Be skeptical of sensational headlines  
        ✅ Verify facts with official sources  
        
        ### 🔮 Future Improvements
        
        - Deep learning models (BERT, LSTM)
        - Source credibility checking
        - Multilingual support
        - Fact-checking API integration
        - Browser extension
        """)
    
    st.markdown("---")
    st.success("📧 **Contact:** For questions or feedback, reach out to your project team!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Fake News Detection System | Built with ❤️ using Streamlit & Machine Learning</p>
        <p>⚠️ Disclaimer: This tool is for educational purposes. Always verify news from multiple sources.</p>
    </div>
    """, unsafe_allow_html=True)
