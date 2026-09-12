# Fake News Detection

An NLP classifier that flags whether a news article is real or fake — built as a capstone project during my data science and analytics program.

## 🎯 Problem
Fake news spreads fast and is hard to catch by eye. The goal here was to build a text classifier that can flag likely-fake articles from content alone — while making sure the model is actually reading the *writing*, not picking up on incidental patterns in how the dataset was assembled.

## 🧠 Approach
- **Model:** TF-IDF vectorization + Logistic Regression, trained on article text only
- **Pipeline:** clean → split → vectorize/scale → SMOTE → feature selection → model → evaluate — in that order, strictly, to avoid data leakage
- **Leakage caught and removed:**
  - The `SUBJECT` column perfectly separated real vs. fake source files — a shortcut that wouldn't hold up on new data
  - A Reuters-specific dateline pattern that appeared only in real articles, letting the model "cheat" by spotting formatting instead of substance
  - Once both were removed, the model was retrained on text-only features to make sure it's actually learning from the writing

## 📊 Results
Model: Logistic Regression
Test F1: 0.9808
Test AUC-ROC: 0.9978

## 🖥️ Demo
Built as a Streamlit app with a dark-themed UI, gauge charts, and verdict cards, plus a PowerPoint deck with real screenshots of the app in action.

**Live app:https://fake-news-detection-project-5srzdq3wurlzgbc3cnpe5a.streamlit.app/

## 🛠️ Tech Stack
Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · Streamlit

## 📁 Structure
```
├── Fake_News_Detection.ipynb   # EDA, pipeline, and model development
├── app.py                      # Streamlit app
├── best_model.joblib           # trained Logistic Regression model
├── tfidf_vectorizer.joblib     # fitted TF-IDF vectorizer
├── chi2_selector.joblib        # feature selector
├── requirements.txt
└── README.md
```
## 🚀 Run it locally
```bash
git clone https://github.com/Alvy-codes/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
streamlit run app.py
```
