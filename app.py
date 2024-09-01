from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load('spam_detection_model.joblib')
feature_names = joblib.load('model_features.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email content from the form
    email_content = request.form['email_content']

    # Preprocess the email content similarly to how the dataset was structured
    email_words = email_content.lower().split()
    
    # Create a word count dictionary with all features
    word_counts = {word: 0 for word in feature_names}

    for word in email_words:
        if word in word_counts:
            word_counts[word] += 1

    # Convert the dictionary to a DataFrame, as expected by the model
    email_df = pd.DataFrame([word_counts])

    # Predict the probability of the email being spam
    spam_probability = model.predict_proba(email_df)[:, 1][0]

    # Determine the result based on a threshold
    if spam_probability > 0.5:
        result = "Spam"
    else:
        result = "Not Spam"

    # Render the result with the probability
    return render_template('index.html', prediction=f"{result} (Spam Probability: {spam_probability:.2%})")

if __name__ == '__main__':
    app.run(debug=True)

