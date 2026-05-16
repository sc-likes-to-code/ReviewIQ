import streamlit as st
import torch
import re
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Page configuration

st.set_page_config(
    page_title='ReviewIQ - AI Sentiment Analysis',
    page_icon='🧠',
    layout='centered'
)

# Label names

LABEL_COLUMNS = [
    'Positive',
    'Negative',
    'Neutral',
    'Recommended',
    'Not Recommended'
]

# Device configuration

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer

@st.cache_resource
def load_model():

    model_path = './saved_model'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# Text preprocessing

def clean_text(text):

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Prediction function

def predict_sentiment(review_text, tokenizer, model):

    cleaned_text = clean_text(review_text)

    encoded_input = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

    probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    results = {}

    for label, probability in zip(LABEL_COLUMNS, probabilities):

        results[label] = round(
            float(probability) * 100,
            2
        )

    return results

# Determine dominant sentiment

def get_final_sentiment(results):

    filtered_results = {
        key: value
        for key, value in results.items()
        if key not in ['Recommended', 'Not Recommended']
    }

    final_sentiment = max(
        filtered_results,
        key=filtered_results.get
    )

    return final_sentiment

# Load model resources

with st.spinner('Loading trained BERT model...'):

    tokenizer, model = load_model()

# Main UI

st.title('🧠 ReviewIQ')

st.markdown(
    '''
    AI-powered sentiment intelligence platform for analyzing
    reviews, opinions, and customer feedback using a
    Transformer-based BERT model.
    '''
)

# System information

col1, col2 = st.columns(2)

with col1:
    st.info(f'Device: {DEVICE}')

with col2:
    st.info('Model: BERT Transformer')

# Review input

review_text = st.text_area(
    'Enter Review or Opinion Text',
    placeholder='Type any review, opinion, or feedback here...',
    height=180
)

# Predict button

predict_button = st.button(
    'Analyze Sentiment',
    use_container_width=True
)

# Prediction section

if predict_button:

    if len(review_text.strip()) == 0:

        st.warning('Please enter a valid review.')

    else:

        with st.spinner('Analyzing review sentiment...'):

            results = predict_sentiment(
                review_text,
                tokenizer,
                model
            )

            final_sentiment = get_final_sentiment(results)

        st.success('Analysis Complete')

        # Final sentiment

        st.subheader('Final Sentiment')

        if final_sentiment == 'Positive':

            st.success(final_sentiment)

        elif final_sentiment == 'Negative':

            st.error(final_sentiment)

        else:

            st.warning(final_sentiment)

        # Confidence scores

        st.subheader('Confidence Scores')

        for label, probability in results.items():

            st.write(f'**{label}** : {probability}%')

            st.progress(
                min(probability / 100, 1.0)
            )

        # Detailed results table

        st.subheader('Detailed Prediction Results')

        results_df = pd.DataFrame({
            'Label': list(results.keys()),
            'Confidence (%)': list(results.values())
        })

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

# Sidebar

st.sidebar.title('About ReviewIQ')

st.sidebar.markdown(
    '''
    ### Features

    - Real-time sentiment analysis
    - Transformer-based BERT inference
    - GPU acceleration with CUDA
    - Confidence probability visualization
    - Multi-label sentiment classification
    - AI-powered review intelligence

    ### Supported Sentiments

    - Positive
    - Negative
    - Neutral
    - Recommended
    - Not Recommended
    '''
)

st.sidebar.success('Model Loaded Successfully')

st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Powered by a Transformer-based BERT model trained on
        34K+ real-world review samples.
    </div>
    """,
    unsafe_allow_html=True
)