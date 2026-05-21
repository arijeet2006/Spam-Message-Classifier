# 📱 Intelligent SMS Spam Classifier Suite

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Natural Language Processing](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning and Natural Language Processing (NLP) infrastructure engineered to analyze, process, and classify mobile text messages. Built from scratch on raw SMS communication data, this system translates chaotic text arrays into pristine numerical vectors, runs probabilistic statistical classification, and surfaces real-time predictions through an intuitive, interactive Streamlit production dashboard.

---

## 📌 Project Overview & Intent

In modern telecommunications, short-message spam presents a continuous security and optimization problem. This project serves as an end-to-end solution mapping out:
1. **Data Sanitization & Structuring:** Turning raw, inconsistent corporate/personal SMS logs into balanced tabular assets.
2. **Feature Engineering & Behavioral Analytics:** Isolating distinct linguistic profiles (character bounds, word clusters, and sentence limits) that distinguish safe communication channels from adversarial tracking hooks.
3. **Probabilistic Class Extraction:** Minimizing False-Positive events using tuned Bayesian classification models to ensure legitimate messages are never accidentally filtered.

---

## 🚀 Key Framework Architecture

### 1. Robust Natural Language Processing Pipeline
Raw text arrays vary widely in spelling, grammar, and syntax. To capture the actual predictive semantics of each text, user inputs undergo an explicit multi-stage pipeline using the **Natural Language Toolkit (NLTK)**:
* **Lowercasing:** Converts all inputs to lowercase to normalize tokens (e.g., matching `SPAM`, `Spam`, and `spam` identically).
* **Word Tokenization:** Parses string structures into concrete word units and punctuation boundaries using `nltk.word_tokenize`.
* **Alphanumeric Scrubbing:** Scrubs text by discarding mathematical syntax variables, specialized structural punctuation icons, and stray characters.
* **Stopword Purging:** Dynamically drops recurring grammatical placeholders (e.g., "our", "have", "you") matching the standard NLTK English stopword array.
* **Porter Stemming Translation:** Normalizes structural suffixes down to core morphological roots via a `PorterStemmer` instance (e.g., transforming `winning`, `wins`, and `winner` to `win`).

### 2. Behavioral Exploratory Data Analysis (EDA) Insights
During model engineering inside `1.ipynb`, advanced custom structural features were calculated to discover text density correlations:
* **Metrics Injected:** `num_characters`, `num_words`, and `num_sentences`.
* **The Structural Contrast Rule:** Statistical distribution mapping proves that safe messages (**Ham**) follow localized, compact spatial distributions (typically under 50 characters). Adversarial structures (**Spam**) heavily scale toward higher character and word density limits, packing sentences with sensory text hooks like "FREE", "CLAIM", and "CASH WINNER".

---

## ⚠️ Challenges Faced & Engineering Solutions

Building an end-to-end NLP product involves overlapping data engineering complexities. Below are the core roadblocks encountered during development and how they were systematically resolved:

### 1. Structural Schema Inconsistencies & Dataset Cleaning
* **The Problem:** The initial raw dataset was plagued with structural parsing errors. Unnamed, corrupt background arrays (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) saturated the index layout, while arbitrary column headers threw off feature mapping. 
* **The Solution:** Implemented a targeted pandas drop operation to isolate and purge the corrupted columns (`df.drop(...)`). Followed this by explicitly locking down the operational parameters to strict descriptors (`label` and `message`) to maintain clean data flow through the tokenization stage.

### 2. High Class-Imbalance Friction
* **The Problem:** Exploratory plotting revealed an overwhelming data distribution disparity: legitimate messages (**Ham**) significantly outnumbered malicious instances (**Spam**). Training standard models on such highly skewed distributions forces class classifiers to over-index on the majority class, leading to severe blindness toward spam signals.
* **The Solution:** Leveraged an explicit data preprocessing strategy paired with a specialized **Bernoulli Naive Bayes** setup. By optimizing the decision engine around boolean text occurrences (word presence vs. absence flag arrays) rather than word volume frequency, the model built strict sensitivity limits for localized spam patterns despite the skewed data ratio.

---

## 🛠️ Mathematical Model Selection & Benchmarks

The project shifts from experimental code to absolute production deployment by training and evaluating three distinct structural branches of Naive Bayes Classifiers using **Bag of Words (CountVectorizer)** feature extractions:

| Engine | Global Accuracy | Precision Score | Validation Performance Profile |
| :--- | :--- | :--- | :--- |
| **Bernoulli Naive Bayes (`BernoulliNB`)** | **97.00%** | **97.34%** | 🏆 **Production Winner:** Unmatched precision profile. |
| **Multinomial Naive Bayes (`MultinomialNB`)** | 96.42% | 83.44% | **Sub-optimal:** Prone to high False-Positive rates. |
| **Gaussian Naive Bayes (`GaussianNB`)** | 88.00% | 53.15% | **Failed Standard:** Struggles with sparse matrix arrays. |

### 🔍 Operational Logic for Bernoulli Selection
While standard text mining engines default to Multinomial tracking, the `BernoulliNB` architecture thrives here. Because SMS structures are brief, tracking whether key risk indicators exist or do not exist proves mathematically sounder than calculating arbitrary frequency distributions. 

Most importantly, it maximizes **Precision**. In production filters, marking a critical personal message as spam (a False Positive) causes major system friction. Our Bernoulli selection ensures that when a message is tagged as **🚨 SPAM**, the indicator is definitive.

---

## 📁 Repository Directory Maps

```text
├── 1.ipynb            # Experimental lab space (Data cleaning, NLTK analysis, EDA plots, Model Benchmarks)
├── app.py             # Active production web script orchestrating Streamlit frontend and caching mechanics
├── bnb_model.pkl      # Serialized Production Bernoulli Naive Bayes Classifier model 
├── vectorizer.pkl     # Serialized Production CountVectorizer sparse-matrix text transformer
└── README.md          # Project documentation suite
