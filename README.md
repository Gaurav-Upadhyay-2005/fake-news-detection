# 📰 TruthLens — AI-Powered Fake News Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NewsAPI](https://img.shields.io/badge/NewsAPI-000000?style=for-the-badge&logo=rss&logoColor=white)

**An end-to-end fake news detection system combining classical ML, Deep Learning (BiLSTM), and Transformer (DistilBERT) models — with a live AI-verified news website.**

*B.Sc. Data Science Final Year Project | Presented at Aavishkar Inter-Collegiate Research Convention, University of Mumbai (2025–26)*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Datasets Used](#-datasets-used)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [ML Models](#-ml-models-nlp-pipeline)
- [Deep Learning Models](#-deep-learning-models)
- [Model Accuracy Summary](#-model-accuracy-summary)
- [DistilBERT Inference Pipeline](#-distilbert-inference-pipeline)
- [How to Run Locally](#-how-to-run-locally)
- [Tech Stack](#-tech-stack)
- [Downloads](#-datasets--saved-models-google-drive)
- [Author](#-author)

---

## 🧠 Project Overview

**TruthLens** is a complete fake news detection system built from scratch — from raw data collection and cleaning, to model training, to deployment as a working web application.

The project has **two separate applications:**

| App | File | Description |
|-----|------|-------------|
| 🖥️ **Streamlit ML Checker** | `new_app.py` | User inputs any news title + content → 3 ML models (LR, SVC, Voting Classifier) predict Real or Fake with confidence score, analytics & PDF sample references |
| 🌐 **Live News Website** | `news_web_page.html` | Real-time news fetched via NewsAPI.org → Every news card has a **"Verify with AI"** button powered by DistilBERT (91% accuracy) |

---

## ✨ Features

### Streamlit App (`new_app.py`)
- 🤖 Choose between **3 trained ML models** from sidebar: Logistic Regression, SVC, Voting Classifier
- 📝 Input any news **title + content** for instant prediction
- 📊 **Analytics tab** with confidence score visualization (Plotly charts)
- 📄 Built-in **PDF references** — sample fake and real news articles for comparison
- ⚙️ Complete preprocessing pipeline shown transparently
- 🎨 Dark-themed professional UI with custom CSS

### Live News Website (`news_web_page.html`)
- 📰 Fetches **live news** from NewsAPI.org across 8 categories: Top Stories, Business, Technology, Science, Health, Sports, Entertainment
- ⚡ **"Verify with AI"** button on every news card
- 🧠 Runs **DistilBERT** (fine-tuned, 91% accuracy) directly in the browser via API
- 📊 Shows **% True / % Fake** confidence score with 60% threshold
- 🔖 Displays preprocessing pipeline used per article

---

## 📁 Project Structure

```
fake-news-detection/
│
├── new_app.py                               # Streamlit ML web app (v3.0)
├── news_web_page.html                       # Live news website with DistilBERT AI verification
├── requirements.txt                         # All Python dependencies
│
├── V2_Data_Cleaning_and_Preprocessing.ipynb # Complete data pipeline (8 datasets merged)
├── V2_NLP_ML_Model_Traning.ipynb            # ML model training (TF-IDF + 6 classifiers)
├── V2__DL_Model_Training.ipynb              # DL training (BiLSTM x2 + DistilBERT)
│
├── fake_news_samples.pdf                    # Reference: sample fake news articles
├── real_news_samples.pdf                    # Reference: sample real news articles
│
├── V2_Saved_Models_ML/                      # Saved ML models (Google Drive ↓)
│   ├── logistic_regression.pkl
│   ├── svc.pkl
│   ├── voting_classifier_86acc.pkl
│   └── tfidf_vectorizer.pkl
│
├── V2_dl_model_saved/                       # Saved DL models (Google Drive ↓)
│   ├── distilbert_m1/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── bi-direct_m1.keras
│   ├── bi-direct_m2.keras
│   └── tokenizer.pkl
│
├── data sets/                               # Raw datasets (Google Drive ↓)
└── V2_cleaned kaggle datasets/              # Final cleaned dataset (Google Drive ↓)
    └── final_dataset.csv
```

---

## 📦 Datasets Used

The final training dataset (`final_dataset.csv`) was created by merging, cleaning, and standardizing **8 different datasets:**

| # | Dataset | Description |
|---|---------|-------------|
| 1 | **Kaggle Fake/Real News** (`fake_.csv`, `True.csv`) | Classic benchmark dataset |
| 2 | **IFND.csv** | Indian Fake News Dataset |
| 3 | **WELFake_Dataset.csv** | Large combined fake news dataset |
| 4 | **LIAR Dataset** (`Final_HuggingFace_LIAR.csv`) | Political statements dataset from HuggingFace |
| 5 | **BharatFakeNews** (`bharatfakenews.xlsx`) | India-specific fake news |
| 6 | **Scraped Fake News** (`Scraped_Fake_News.csv`, `V2_Scraped_Fake_News.csv`) | Custom scraped articles |
| 7 | **API Real News** (`API_Real_News_Large.csv`, `V2_API_Real_News_Large.csv`) | Real news via API |
| 8 | **LIAR TSV** (`train.tsv`, `test.tsv`, `valid.tsv`) | Original LIAR benchmark |

> **Final Dataset Size: 131,163+ articles** | Label: `0 = Real`, `1 = Fake`

---

## 🧹 Data Cleaning & Preprocessing

**Notebook:** `V2_Data_Cleaning_and_Preprocessing.ipynb`

Each dataset was independently cleaned and standardized before merging:

- Renamed columns to unified schema: `title`, `text`, `label`
- Standardized labels to binary: `0 = Real`, `1 = Fake`
- Removed null values and duplicate rows (by `text` column)
- Language filtering to keep only English articles
- Merged all 8 datasets → shuffled → saved as `final_dataset.csv`

---

## 🤖 ML Models (NLP Pipeline)

**Notebook:** `V2_NLP_ML_Model_Traning.ipynb`

### NLP Preprocessing Steps
1. Lowercase conversion
2. Punctuation removal
3. URL and HTML tag removal
4. Emoji and special character removal
5. Stopword removal (NLTK English stopwords)
6. Lemmatization (WordNetLemmatizer)

### TF-IDF Vectorization
```
TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english', max_df=0.9, min_df=5)
```

### Train / Validation / Test Split
```
70% Training | 15% Validation | 15% Testing  (stratified)
```

### Models Trained
| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | **85.50%** |
| SVC (LinearSVC + CalibratedClassifierCV) | **85.60%** |
| Naive Bayes | ~83% |
| Decision Tree | ~82% |
| Random Forest | ~83% |
| XGBoost | ~84% |
| **Voting Classifier (Soft, All 6)** | **86.09%** ✅ Best ML |

> Saved models: `logistic_regression.pkl`, `svc.pkl`, `voting_classifier_86acc.pkl`, `tfidf_vectorizer.pkl`

---

## 🧠 Deep Learning Models

**Notebook:** `V2__DL_Model_Training.ipynb`

### Two Preprocessing Strategies Compared
| Strategy | Description | Used For |
|----------|-------------|----------|
| `basic_clean` | Lowercase + remove URLs/HTML/non-alpha | BiLSTM M2, DistilBERT |
| `advanced_clean` | basic_clean + stopword removal + lemmatization | BiLSTM M1 |

### BiLSTM Model 1 (with NLP)
```
Embedding(10000, 128) → Bidirectional LSTM(64) → Dropout(0.5) → Dense(1, sigmoid)
Epochs: 10 | Batch: 64 | EarlyStopping(patience=2) | MAX_LEN: 300
```

### BiLSTM Model 2 (without NLP — raw text)
```
Embedding(20000, 128) → Bidirectional LSTM(64) → Dropout(0.6) → Dense(1, sigmoid)
Epochs: 6 | Batch: 64 | EarlyStopping(patience=2) | MAX_LEN: 400
```

### DistilBERT (Transformer)
```
Model: distilbert-base-uncased (HuggingFace)
Fine-tuned on: 131,163+ articles
MAX_LEN: 128 | Epochs: 2 | Weight Decay: 0.01
Eval Strategy: per epoch
```

### DL Model Accuracy
| Model | Test Accuracy |
|-------|--------------|
| BiLSTM M1 (with NLP) | ~88% |
| BiLSTM M2 (without NLP) | ~89% |
| **DistilBERT** | **91%** ✅ Best Overall |

---

## 📊 Model Accuracy Summary

| Model | Type | Test Accuracy |
|-------|------|--------------|
| Logistic Regression | ML | 85.50% |
| SVC | ML | 85.60% |
| Voting Classifier | ML Ensemble | 86.09% |
| BiLSTM M1 | Deep Learning | ~88% |
| BiLSTM M2 | Deep Learning | ~89% |
| **DistilBERT** | **Transformer** | **91%** 🏆 |

---

## 🔄 DistilBERT Inference Pipeline

```
News Title + Article Text
         ↓
  content = title + " " + text
         ↓
  basic_clean():
    → lowercase
    → remove URLs (https?://\S+ | www.\S+)
    → remove HTML tags (<.*?>)
    → remove non-alpha chars ([^a-z\s])
         ↓
  DistilBertTokenizerFast
    → padding = max_length
    → MAX_LEN = 128
    → truncation = True
         ↓
  DistilBERT inference → softmax
         ↓
  label_map: { 0 → Real ✅, 1 → Fake ❌ }
  Threshold: 60%
         ↓
  ✅ REAL NEWS  /  ❌ FAKE NEWS  /  ⚠️ UNCERTAIN
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/Gaurav-Upadhyay-2005/fake-news-detection.git
cd fake-news-detection
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download models & datasets from Google Drive
> 📥 [Download from Google Drive](https://drive.google.com/drive/folders/1AcVJHfpAj3p-W5aWteiRsb6s_0r_wUKw?usp=sharing)

After downloading, place them in the project root so the structure looks like:
```
fake-news-detection/
├── V2_Saved_Models_ML/
├── V2_dl_model_saved/
├── data sets/
└── V2_cleaned kaggle datasets/
```

### Step 4 — Run the Streamlit App
```bash
streamlit run new_app.py
```
App will open at `http://localhost:8501`

### Step 5 — Run the Live News Website
- Add your NewsAPI key inside `news_web_page.html`
- Open `news_web_page.html` directly in any browser
- Or serve it locally:
```bash
python -m http.server 8000
# Then open: http://localhost:8000/news_web_page.html
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Language** | Python 3.10+ |
| **Web App** | Streamlit |
| **ML** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras, PyTorch |
| **NLP / Transformers** | HuggingFace Transformers (DistilBERT) |
| **Text Processing** | NLTK (stopwords, lemmatizer, tokenizer) |
| **Vectorization** | TF-IDF (Scikit-learn) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data** | Pandas, NumPy |
| **News API** | NewsAPI.org |
| **Frontend** | HTML, CSS, JavaScript |
| **Model Saving** | Joblib, Pickle, HuggingFace safetensors |

---

## 📥 Datasets & Saved Models (Google Drive)

> GitHub does not support large files (datasets: 444 MB, models: 316 MB). All resources are available on Google Drive.

| Resource | Contents | Link |
|----------|----------|------|
| 📁 **Datasets + Saved Models** | `data sets/`, `V2_cleaned kaggle datasets/`, `V2_Saved_Models_ML/`, `V2_dl_model_saved/` | [🔗 Download Here](https://drive.google.com/drive/folders/1AcVJHfpAj3p-W5aWteiRsb6s_0r_wUKw?usp=sharing) |

---

## 👨‍💻 Author

**Gaurav Upadhyay**
B.Sc. Data Science | CGPA: 9.76 / 10
Bunts Sangha's S.M. Shetty College of Science, Commerce and Management Studies, Mumbai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gaurav-upadhyay-g)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gaurav-Upadhyay-2005)

---

## 📅 Project Timeline

| Milestone | Date |
|-----------|------|
| Project Started | November 2025 |
| Data Collection & Cleaning | November – December 2025 |
| ML Model Training | December 2025 |
| DL & DistilBERT Training | January 2026 |
| Web App Development | January – February 2026 |
| Project Completed | February 2026 |
| Aavishkar Research Convention | 2025–26 |

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

</div>
