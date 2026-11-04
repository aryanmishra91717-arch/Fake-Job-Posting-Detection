import pandas as pd
import joblib

# Load trained model
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Take job details from user
title = input("Enter job title: ")
description = input("Enter job description: ")

# Create input data
data = pd.DataFrame({
    "title": [title],
    "description": [description]
})

# Combine text
data["text"] = data["title"] + " " + data["description"]

# Convert text into features
X = vectorizer.transform(data["text"])
prediction = model.predict(X)

if prediction[0] == 1:
    print("\n⚠️ This job posting may be FRAUDULENT.")
else:
    print("\n✅ This job posting appears to be GENUINE.")