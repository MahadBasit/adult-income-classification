import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('adult_income1.csv')

df['workclass'] = df['workclass'].replace({'?': 'Unknown'})
df['income'] = df['income'].replace({'<=50K': 0, ">50K": 1})
df['income'] = df['income'].astype(int)
df['US_or_Non_US'] = np.where(df['native.country'] == 'United-States', 'US', 'Non-US')
df = df.drop(columns=['occupation', 'education', 'fnlwgt', 'native.country'])
df = pd.get_dummies(df, columns=['workclass', 'sex', 'marital.status', 'race', 'relationship', 'US_or_Non_US'], drop_first=True, dtype=int)

X = df.drop(columns=['income'])
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(model.intercept_)
print("Model training complete!")

predictions = model.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", matrix)

report = classification_report(y_test, predictions)
print("\nClassification Report:\n", report)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")