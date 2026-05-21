# train_and_save.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load and clean data
df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df.drop_duplicates(inplace=True)
df['label_spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(cleaned)

df['processed_message'] = df['message'].apply(transform_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_message']).toarray()
y = df['label_spam'].values

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Train classifier
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

# Save model and vectorizer
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully!")