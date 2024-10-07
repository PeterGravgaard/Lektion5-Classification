import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns

data = pd.read_csv('housing.csv')

# Fjern rækker med manglende værdier i kolonnen 'total_bedrooms' og 'ocean_proximity'
data_cleaned = data.dropna(subset=['total_bedrooms'])
data_cleaned = data_cleaned.drop(columns=['ocean_proximity'])
data_cleaned['rooms_per_household'] = data_cleaned['total_rooms'] / data_cleaned['households']
data_cleaned['bedrooms_per_room'] = data_cleaned['total_bedrooms'] / data_cleaned['total_rooms']
data_cleaned['population_per_household'] = data_cleaned['population'] / data_cleaned['households']


# Beregn Mean Absolute Percentage Error (MAPE)
def  accurazy(y_test, y_pred):
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    accuracy = 100 - mape
    print(f"Accuracy: {accuracy:.2f}%")

def visualiser_pred(X_test, y_pred):
    plt.scatter(X_test['longitude'], X_test['latitude'], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
    plt.colorbar(label='Forudsete huspriser')
    plt.xlabel('Længdegrad')
    plt.ylabel('Breddegrad')
    plt.title('Forudsete huspriser baseret på lokation')
    plt.show()

def visualiser_confusion_matrix(y_test, y_pred):
    # Sørg for, at de forudsete værdier er ikke-negative
    y_pred = np.clip(y_pred, 0, None)
    
    # Opdel huspriser i kategorier (fx 3 kategorier: lav, middel, høj)
    bins = [0, 150000, 300000, np.inf]
    labels = ['Lav', 'Middel', 'Høj']

    # Klassificér de faktiske og forudsete huspriser i kategorier
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)

    # Sørg for, at både y_test_binned og y_pred_binned er serier med samme type
    y_test_binned = y_test_binned.astype(str)
    y_pred_binned = y_pred_binned.astype(str)

    # Beregn confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)

    # Visualisér confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Forudsete kategorier')
    plt.ylabel('Faktiske kategorier')
    plt.title('Confusion Matrix for Huspris Kategorier')
    plt.show()

# Vælg de relevante kolonner (longitude, latitude, median_house_value)
        #   X = data_cleaned[['longitude', 'latitude']]
X = data_cleaned[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                  'total_bedrooms', 'population', 'households', 'median_income', 
                  'rooms_per_household', 'bedrooms_per_room', 'population_per_household']]
y = data_cleaned['median_house_value']

# Del data op i trænings- og testdatasæt (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opret en lineær regressionsmodel
model = LinearRegression()

# Træn modellen på træningsdatasættet
model.fit(X_train, y_train)

# Forudsig huspriser på testdatasættet
y_pred = model.predict(X_test)

# Beregn fejl (mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#visualiser_pred(X_test, y_pred)
#visualiser_confusion_matrix(y_test, y_pred)
#data = data.info()
print(data_cleaned.info())
accurazy(y_test, y_pred)