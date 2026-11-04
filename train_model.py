import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading cleaned dataset...")

# Load cleaned dataset
df = pd.read_csv("cleaned_job_postings.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Combine important text columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

df["text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

# Input and target
X = df["text"]
y = df["fraudulent"]

print("Preparing data...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# Convert text into numbers
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text converted into numerical features.")

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "fake_job_model.pkl")

print("Model trained successfully!")

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Detailed results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))