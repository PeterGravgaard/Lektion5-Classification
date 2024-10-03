import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


'### 1. LOAD CSV FILEN OG FJERN INDEKSERINGSKOLONNER ###'
df = pd.read_csv('breast-cancer_filtered.csv')
df = df.drop('Unnamed: 0', axis=1)
print(df.info())

'### 2. SCATTERPLOT UD FRA UDVALGTE FEATURES ###'
df.plot(kind="scatter",x='radius_mean', y='texture_mean', c='diagnosis', cmap='jet', colorbar=True)

'### 3. VÆLG EN BINÆR KLASSIFIER UD FRA BOGEN ###'
model = LogisticRegression()

'### 4. KLARGØR DATA ###'
input = df.drop('diagnosis', axis=1)
output = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.20, random_state=42)

'## TRÆNING AF MODEL ##'
model.fit(X_train, y_train)

'### 5. LAV EN CONFUSION MATIRX ###'
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted B', 'Predicted M'],
            yticklabels=['Actual B', 'Actual M'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


'### 6. LAV EN PLOT AF EN ROC-KURVE ###'
pred_proba = model.predict_proba(X_test)
pred_proba = pred_proba[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.show()
