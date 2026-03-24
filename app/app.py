import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Complaint Classification")

df = pd.read_csv("data/processed/cleaned_data.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

user_input = st.text_area("Enter a complaint")

if st.button("Predict"):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        st.success(f"Predicted category: {prediction}")
    else:
        st.warning("Please enter some text.")