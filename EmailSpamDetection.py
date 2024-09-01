# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('C:/Users/Hp/OneDrive/Documents/PersonalProjects/EmailSpamDetectionApp/Data/emails.csv')

ham_msg = data[data.Prediction == 0]  # Ham emails (non-spam)
spam_msg = data[data.Prediction == 1]  # Spam emails

# Upsample the spam messages to match the number of ham messages
spam_msg = spam_msg.sample(n=len(ham_msg), replace=True, random_state=42)

# Combine the upsampled spam messages with the ham messages
balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)

# Drop the 'Prediction' and 'Email No.' columns from the feature set
X = balanced_data.drop(columns=['Prediction', 'Email No.'])

# Target variable
y = balanced_data['Prediction']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting the target for test data
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


joblib.dump(model, 'spam_detection_model.joblib')
joblib.dump(X.columns, 'model_features.joblib')