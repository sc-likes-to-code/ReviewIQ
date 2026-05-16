# 🧠 ReviewIQ

AI-powered sentiment intelligence platform built using BERT, PyTorch, CUDA, and Streamlit for real-time multi-label sentiment analysis.

---

# 🚀 Overview

ReviewIQ is a GPU-accelerated NLP application capable of analyzing reviews, opinions, and customer feedback using a Transformer-based BERT model.

The model performs multi-label sentiment classification and predicts:

- Positive
- Negative
- Neutral
- Recommended
- Not Recommended

The application supports real-time inference through an interactive Streamlit web interface with confidence probability visualization.

---

# ✨ Features

- Transformer-based BERT sentiment analysis
- Multi-label sentiment classification
- Real-time review prediction
- Confidence probability visualization
- GPU acceleration with CUDA
- Interactive Streamlit web application
- Saved model inference pipeline
- Clean and professional UI
- Support for multiple review domains

---

# 🧠 Model Details

- Model Architecture: BERT (bert-base-uncased)
- Framework: PyTorch
- Interface: Streamlit
- Training Device: NVIDIA RTX 4060 Laptop GPU
- Mixed Precision Training Enabled
- Multi-label classification using sigmoid activation

The model was trained on 34K+ real-world Amazon product reviews and generalized for broader sentiment analysis tasks.

---

# 📊 Model Performance

## Best Test Set Results

| Metric | Score |
|---|---|
| Test Accuracy | 92.87% |
| Test F1 Micro | 96.00% |
| Test F1 Macro | 72.58% |
| Test Precision Micro | 96.24% |
| Test Recall Micro | 95.76% |

## Training Performance Summary

The model was trained and evaluated across multiple training runs to verify consistency and stability.

Key observations:

- Stable convergence across training sessions
- Strong generalization performance
- Consistent validation accuracy around 92%+
- Effective multi-label sentiment prediction
- High precision and recall performance

---

# 🖥️ Application Screenshots

## Main Interface

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/92f4e6c8-c7c0-4726-95eb-2773659804e1" />


---

## Positive Review Prediction

<img width="1919" height="1079" alt="Screenshot 2026-05-17 013253" src="https://github.com/user-attachments/assets/668b90e2-2a3b-4f9a-8c25-45483a230728" />
<img width="1919" height="1079" alt="Screenshot 2026-05-17 013259" src="https://github.com/user-attachments/assets/77348a42-d782-43ab-af1d-8e5302018911" />


---

## Negative Review Prediction

<img width="1919" height="1079" alt="Screenshot 2026-05-17 013323" src="https://github.com/user-attachments/assets/27fc14d5-2d24-44cd-9194-25d5c903b6ce" />
<img width="1919" height="1079" alt="Screenshot 2026-05-17 013328" src="https://github.com/user-attachments/assets/4ee86b2e-36ae-4654-b6db-b6d84af6bff0" />


---

# 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- CUDA
- NVIDIA GPU Acceleration
- Pandas
- Scikit-learn

---

# 📂 Project Structure

```text
ReviewIQ/
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data.csv
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ReviewIQ
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run The Application

## Launch Streamlit App

```bash
streamlit run app.py
```

---

# 🧪 Train The Model

```bash
python train.py
```

---

# 💾 Model Weights

The trained BERT model weights are not included in this repository due to file size limitations.

You can train the model locally using:

```bash
python train.py
```

---

# 📌 Future Improvements

- CSV batch review prediction
- Advanced analytics dashboard
- Exportable prediction reports
- Deployment support
- Custom frontend integration
- Aspect-based sentiment analysis

---

# 👨‍💻 Author

Developed as an advanced NLP and AI engineering project focused on Transformer-based sentiment intelligence and real-time inference systems.
