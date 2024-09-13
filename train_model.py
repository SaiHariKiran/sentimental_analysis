import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=column_names)

# Keep only necessary columns
data = data[['text', 'target']]

# Map sentiments to 1 (positive) and 0 (negative)
data['target'] = data['target'].map({4: 1, 0: 0})

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)    # remove mentions
    text = re.sub(r"#\w+", "", text)    # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['text'] = data['text'].apply(preprocess_text)

X = data['text']
y = data['target']

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
