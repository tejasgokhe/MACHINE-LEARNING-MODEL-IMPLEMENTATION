import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#LOAD THE DATASET
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=["label", "message"])

#DATA PREPROCESSING
df['message'] = df['message'].apply(lambda x: x.lower())  #Convert text to lowercase
X = df['message'] 
y = df['label'].map({'ham': 0, 'spam': 1}) 

#SPLIT DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#CONVERT TEXT DATA INTO NUMERIC DATA
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#TRAIN THE MODEL
model = MultinomialNB()
model.fit(X_train_vec, y_train)

#EVALUATING MODEL
y_pred = model.predict(X_test_vec)

#ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#CLASSIFICATION REPORT
print(classification_report(y_test, y_pred))

#INCASE NEED TO USE IN FUTURE
import joblib
#TO SAVE MODEL
joblib.dump(model, 'spam_classifier_model.pkl')
#TO LOAD THE MODEL
loaded_model = joblib.load('spam_classifier_model.pkl')
