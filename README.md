# 📱 SMS Spam Classifier Suite

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Natural Language Processing](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Natural Language Processing (NLP) framework designed to identify and classify spam text messages. Utilizing explicit lexical tokenization, stopword clearing, and morphological stemming pipelines combined with a optimized Naive Bayes decision algorithm, this repository yields accurate indicators to map incoming communication into either **🚨 SPAM** or **✅ HAM** classifications via an intuitive production-ready Streamlit dashboard interface.

---

## 🚀 Key Features

* **Advanced Text Preprocessing Engine:** Implements deep string cleaning pipelines through the Python Natural Language Toolkit (NLTK):
  * **Case Normalization & Tokenization:** Forces lowercase structures and strips content down into distinct structural lexical tokens.
  * **Alphanumeric Noise Filtering:** Purges mathematical operators, symbols, and stray punctuation characters.
  * **Lexicon Stopword Removal:** Discards highly recurring contextual non-descriptors (e.g., "is", "the", "at").
  * **Porter Stemming Algorithm:** Reduces inflectional variations down to single baseline root stems.
* **Granular Exploratory Analytics (EDA):** Notebook implementations that engineer structural metrics track tracking character counts, word density, and punctuation weights over true text lengths.
* **Engineered Naive Bayes Benchmarks:** Features a multi-model training layer pitting Gaussian, Multinomial, and Bernoulli variants against each other to optimize model precision.
* **Production Web Dashboard:** Features a decoupled Streamlit web user interface utilizing caching decorators to compile and stream rapid predictive classifications locally.

---

## 🛠️ Model Performance Benchmarks

Based on the empirical experimental outcomes using the **Bag of Words (CountVectorizer)** feature transformation on a 20% validation split, here are the statistical accuracy profiles:

| Algorithm Classification Engine | Global Accuracy | Precision Score | Primary Metric Status |
| :--- | :--- | :--- | :--- |
| **Bernoulli Naive Bayes (`BernoulliNB`)** | **97.00%** | **97.34%** | 🏆 **Production Standard Selected** |
| **Multinomial Naive Bayes (`MultinomialNB`)** | 96.42% | 83.44% | High Recall Bias |
| **Gaussian Naive Bayes (`GaussianNB`)** | 88.00% | 53.15% | High Noise Profile |

> **Operational Insight:** In true spam filtering systems, **Precision is the critical metric** because blocking a legitimate message (a False Positive "Ham" classified as "Spam") causes major communication failure. The Bernoulli Naive Bayes model minimizes this tracking penalty perfectly.

---

## 📁 Repository Structure

```text
├── 1.ipynb            # Experimental workspace (EDA, Feature Extraction, Model Benchmarks)
├── app.py             # Streamlit application orchestration source code
├── bnb_model.pkl      # Production Serialized Bernoulli Naive Bayes Classifier instance
├── vectorizer.pkl     # Production Serialized CountVectorizer Feature Transformer 
└── README.md          # Comprehensive Project Documentation
