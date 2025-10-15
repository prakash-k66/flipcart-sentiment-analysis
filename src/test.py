# src/test.py
import pickle
from cleaning import clean_text

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def predict_review(review):
    """
    Predict sentiment for a single review
    """
    review_cleaned = clean_text(review)
    review_vector = tfidf.transform([review_cleaned])
    return model.predict(review_vector)[0]

# Example usage
if __name__ == "__main__":
    while True:
        review = input("Enter a Flipkart review (type 'exit' to quit): ")
        if review.lower() == "exit":
            print("Goodbye")
            break
        print("Predicted Sentiment:", predict_review(review))
