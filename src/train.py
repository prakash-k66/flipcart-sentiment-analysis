# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from cleaning import clean_text  # import cleaning function

# Replace with the exact file name
data = pd.read_csv("Dataset-SA.csv")


# Clean review text
data['cleaned_review'] = data['Review'].apply(clean_text)

# Convert Rate to numeric
data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce')

# Assign sentiment
def sentiment(rate):
    if pd.isna(rate):
        return "Neutral"
    elif rate >= 4:
        return "Positive"
    elif rate == 3:
        return "Neutral"
    else:
        return "Negative"

data['Sentiment'] = data['Rate'].apply(sentiment)

# Prepare features and target
X = data['cleaned_review']
y = data['Sentiment']

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_vectorized = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved in models/ folder")
