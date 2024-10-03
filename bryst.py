import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


'### 1. LOAD CSV FILEN OG FJERN INDEKSERINGSKOLONNER ###'
df = pd.read_csv('breast-cancer_filtered.csv')
df = df.drop('Unnamed: 0', axis=1)

#Valg af features
feature_x = 'radius_mean'
feature_y = 'texture_mean'

#Opdel datasættet i test og træn
X = df[[feature_x, feature_y]]
y = df['diagnosis']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)

def scatter_plot(feature_x, feature_y, df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df['diagnosis'], palette='Set1')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title='Diagnose', labels=['Godartet', 'Ondsindet'])
    plt.show()

def scatter_plot_with_preditions(feature_x, feature_y, X_test, y_pred):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=X_test[feature_x], y=X_test[feature_y], hue=y_pred, palette={0: 'blue', 1: 'red'}, legend=False)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title='Diagnose', labels=['Godartet', 'Ondsindet'])
    plt.title('Forudsigelser fra logistisk regression')
    plt.show()
   
def missing_data(df):
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])

def plot_histogram(df):
    df.hist(bins=30, figsize=(20, 15))
    plt.show()
     
def confusion_matrix_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Godartet', 'Ondsindet'],
                yticklabels=['Godartet', 'Ondsindet'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def sensitivity_specificity(y_test, y_pred):
    cm=confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    #Udregn sensitivity og specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    
def precision_recall(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print('Precision:', precision)
    print('Recall:', recall)

#Træner på model
model = LogisticRegression()
model.fit(X_train, y_train)

# Gætte tid
y_pred = model.predict(X_test)

#confusion_matrix_plot(y_test, y_pred)
#plot_histogram(df)
#scatter_plot(feature_x, feature_y, df)
#scatter_plot_with_preditions(feature_x, feature_y, X_test, y_pred)
#print(df.info())
precision_recall(y_test, y_pred)
sensitivity_specificity(y_test, y_pred)

""" #Anvend log-transformation på en feature for at håndtere skæv fordeling
    df['feature_name'] = np.log1p(df['feature_name'])
"""