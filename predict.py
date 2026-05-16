import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
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
print(f'Using device: {DEVICE}')

# Saved model path

MODEL_PATH = './saved_model'

# Load tokenizer

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load trained model

print('Loading trained model...')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to GPU/CPU

model.to(DEVICE)
model.eval()

print('Model loaded successfully.\n')

# Text cleaning function

def clean_text(text):

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Prediction function

def predict_sentiment(review_text):

    # Clean input text

    cleaned_text = clean_text(review_text)

    # Tokenize input

    encoded_input = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    # Move tensors to device

    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    # Perform inference

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

    # Convert logits to probabilities

    probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    # Store predictions

    prediction_results = {}

    for label, probability in zip(LABEL_COLUMNS, probabilities):

        prediction_results[label] = round(
            float(probability) * 100,
            2
        )

    return prediction_results

# Display predictions

def display_predictions(results):

    print('\nSentiment Prediction Results')
    print('-' * 45)

    for label, probability in results.items():

        status = (
            'Detected'
            if probability >= 50
            else 'Not Detected'
        )

        print(f'{label}: {probability}% ({status})')

# Main application loop

def main():

    print('Amazon Review Sentiment Predictor Ready')
    print("Type 'exit' to stop the program.\n")

    while True:

        user_review = input('Enter a review: ')

        # Exit condition

        if user_review.lower() == 'exit':

            print('\nExiting predictor...')
            break

        # Empty input check

        if len(user_review.strip()) == 0:

            print('Please enter a valid review.\n')
            continue

        # Generate predictions

        prediction_results = predict_sentiment(user_review)

        # Display output

        display_predictions(prediction_results)

        print('\n' + '=' * 55 + '\n')

# Run application

if __name__ == '__main__':
    main()