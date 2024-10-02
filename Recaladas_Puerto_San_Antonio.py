#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:21:38 2024

@author: momos_macbook_pro
"""

# Required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import openpyxl as openpyxl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import warnings

# 1. Load / Cleaning / Manipulation of the dataframe

datos_recaladas = pd.read_excel("Recaladas_Puerto_San_Antonio.xlsx")

datos_recaladas.info()
datos_recaladas.describe()
datos_recaladas.columns

# Date range
print("Date range:", datos_recaladas['FECHA EFECTIVA'].min(), "to", datos_recaladas['FECHA EFECTIVA'].max())

# Convert date columns to datetime format without/hours
datos_recaladas['FECHA_ESTIMADA_DIA']= pd.to_datetime(datos_recaladas['FECHA ESTIMADA'].dt.strftime('%Y-%m-%d'))
datos_recaladas['FECHA_EFECTIVA_DIA']= pd.to_datetime(datos_recaladas['FECHA EFECTIVA'].dt.strftime('%Y-%m-%d'))

# Create new columns: Cumple, Dias_Atraso, Adelanto, Diferencia
datos_recaladas['Cumple'] = np.where(datos_recaladas['FECHA_ESTIMADA_DIA'] >= datos_recaladas['FECHA_EFECTIVA_DIA'], 1, 0)
datos_recaladas['Dias_Atraso'] = np.where(datos_recaladas['Cumple'] == 1, 0, (datos_recaladas['FECHA_EFECTIVA_DIA'] - datos_recaladas['FECHA_ESTIMADA_DIA']).dt.days)
datos_recaladas['Adelanto'] = np.where(datos_recaladas['FECHA_ESTIMADA_DIA'] > datos_recaladas['FECHA_EFECTIVA_DIA'], 1, 0)
datos_recaladas['Dias_Adelanto'] = np.where(datos_recaladas['Adelanto'] == 1, (datos_recaladas['FECHA_ESTIMADA_DIA'] - datos_recaladas['FECHA_EFECTIVA_DIA']).dt.days, 0)
datos_recaladas['Diferencia'] = (datos_recaladas['FECHA_EFECTIVA_DIA'] - datos_recaladas['FECHA_ESTIMADA_DIA']).dt.days

# Drop NA values

datos_recaladas.isna().sum()
datos_recaladas_limpio = datos_recaladas.dropna() #We are going to drop the observations without Site

# 2. Histogram and Outlier Analysis

sns.histplot(datos_recaladas_limpio['Diferencia'], bins=50, color='darkblue', kde=False)
plt.title('Histograma de Diferencia días Fecha Estimada vs Fecha Efectiva')
plt.xlabel('Diferencia (días) - [Negativo(Adelantos)/Positivo(Atrasos)]')
plt.ylabel('Frecuencia')
plt.show()

# Identifying Outliers using IQR
Q1 = datos_recaladas_limpio['Diferencia'].quantile(0.25)
Q3 = datos_recaladas_limpio['Diferencia'].quantile(0.75)
IQR = Q3 - Q1
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

# Filter out the outliers / We identify a better result in the model with the outliers
# datos_recaladas_limpio = datos_recaladas_limpio[(datos_recaladas_limpio['Diferencia'] >= lower_threshold) & (datos_recaladas_limpio['Diferencia'] <= upper_threshold)]

# 3. Clustering for AGENCIA and SITIO

def cluster_agencia(agencia):
    if agencia in ["AGENCIAS MARITIMAS AGENTAL LIMITADA", "AGENTAL"]:
        return "AGENTAL"
    elif agencia == "AGENCIAS UNIVERSALES S.A.":
        return "AGENCIAS UNIVERSALES"
    elif agencia in ["B & M AGENCIA MARITIMA S.A.", "B&M"]:
        return "B & M AGENCIA MARITIMA"
    elif agencia == "AGUNSA":
        return "AGUNSA"
    elif agencia == "BROOM":
        return "BROOM"
    elif agencia in ["EVERGREEN", "EVERGREEN SHIPPING AGENCY (CHILE)", "EVERGREEN SHIPPING AGENCY ( CHILE )"]:
        return "EVERGREEN"
    elif agencia in ["IAN TAYLOR Y CIA. S.A.","TAYLOR"]:
        return "IAN TAYLOR"
    elif agencia in ["INCHCAPE", "INCHCAPESHIPPING SERVICES B.V. CHILE LIMITADA"]:
        return "INCHCAPE"
    elif agencia == "MARVAL":
        return "MARVAL"
    elif agencia in ["MEDITERRANEAN SHIPPING COMPANY (CHILE) S.A.", "MSC"]:
        return "MSC"
    elif agencia == "MTA":
        return "MTA"
    elif agencia in ["SOCIEDAD MARITIMA Y COMERCIAL SOMARCO LTDA.", "SOMARCO"]:
        return "SOMARCO"
    elif agencia in ["ULTRAMAR", "ULTRAMAR AGENCIA MARITIMA LIMITADA"]:
        return "ULTRAMAR"
    else:
        return "Otros"

datos_recaladas_limpio['Cluster_AGENCIA'] = datos_recaladas_limpio['AGENCIA'].apply(cluster_agencia)

df_agencias = datos_recaladas_limpio.groupby(["AGENCIA", "Cluster_AGENCIA"]).agg(servicios=("AGENCIA", "count"))

### Agency service KPI analysis

cumple_agencia = datos_recaladas_limpio.groupby("Cluster_AGENCIA").agg(
    cumple_total = ("Cumple",np.sum),
    servicios = ("Cumple","count")).sort_values(by = "cumple_total", ascending = False)

cumple_agencia["porc_cumple"] = cumple_agencia["cumple_total"]/cumple_agencia["servicios"]

cumple_agencia = cumple_agencia.sort_values(by = "porc_cumple", ascending = False)



### We identify a significant variance in the target variable between every Agency

## Clustering for Sitios de Recalada

def assign_cluster_sitios(sitio):
    if 'DP World' in sitio:
        return 'DP World'
    elif 'EPSA' in sitio:
        return 'EPSA'
    elif 'PANUL' in sitio:
        return 'PANUL'
    elif 'PUERTO DE SAN ANTONIO' in sitio:
        return 'Puerto de San Antonio'
    elif 'QC' in sitio:
        return 'QC'
    elif 'STI' in sitio or 'SAN ANTONIO TERMINAL INTERNACIONAL' in sitio:
        return 'STI (Terminal Internacional)'
    elif 'TERMINAL COSTANERA' in sitio:
        return 'Terminal Costanera'
    elif 'VOPAK' in sitio:
        return 'VOPAK'
    else:
        return 'Unknown'  # For cases where the SITIO is not recognized

datos_recaladas_limpio['Cluster_SITIOS'] = datos_recaladas_limpio['SITIO'].apply(assign_cluster_sitios)

### Site service KPI analysis

cumple_sitios = datos_recaladas_limpio.groupby("Cluster_SITIOS").agg(
    cumple_total = ("Cumple",np.sum),
    servicios = ("Cumple","count")).sort_values(by = "cumple_total", ascending = False)

cumple_sitios["porc_cumple"] = cumple_sitios["cumple_total"]/cumple_sitios["servicios"]

cumple_sitios = cumple_sitios.sort_values(by = "porc_cumple", ascending = False)

### We identify a significant variance in the Cumple target between every Site

### Adding other date attributes prior correlation analysis

datos_recaladas_limpio['mes_estimado'] = datos_recaladas_limpio['FECHA ESTIMADA'].dt.month
datos_recaladas_limpio['dia_estimado'] = datos_recaladas_limpio['FECHA ESTIMADA'].dt.day
datos_recaladas_limpio['weekday'] = datos_recaladas_limpio['FECHA ESTIMADA'].dt.strftime('%A')  # Full name of the weekday

# Create 'Cluster_Dia' column based on the day of the month
def categorize_day(dia):
    if 1 <= dia <= 10:
        return 'Inicio Mes'
    elif 11 <= dia <= 20:
        return 'Medio Mes'
    elif 21 <= dia <= 31:
        return 'Final Mes'
    else:
        return 'Desconocido'  # Safety for out-of-range values

datos_recaladas_limpio['Cluster_Dia'] = datos_recaladas_limpio['dia_estimado'].apply(categorize_day)

# Perform a left join (merge) on 'FECHA_ESTIMADA_DIA' from 'datos_recaladas_limpio' and 'date' from 'datos_meteorologicos_final'
datos_recaladas_limpio = pd.merge(
    datos_recaladas_limpio,
    datos_meteorologicos_final,
    left_on='FECHA_ESTIMADA_DIA',
    right_on='date',
    how='left'
)

datos_recaladas_limpio.info()

# Rename the column 'INGRESO O SALIDA' to 'Ingreso_Salida'
datos_recaladas_limpio = datos_recaladas_limpio.rename(columns={'INGRESO O SALIDA': 'Ingreso_Salida'})

### Now we have a dataframe with all the atributes to start making some correlation analysis

# 4. Análisis de correlación de variables independientes con el Atributo Cumple nivel de servicio

# Select relevant columns for correlation analysis
corr_df = datos_recaladas_limpio[["mes_estimado","dia_estimado","tavg","prcp","wdir","wspd","Lluvia","Viento_Fuerte","Cumple"]]

# Compute the correlation matrix (only for continuous/numeric variables)
correlation = corr_df.corr()

# Plot the correlation matrix using seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
plt.title('Correlation Matrix')
plt.show()

### With these continuous variables, we do not find any correlation with the target variable.

# 5. Model Building (Training and Testing Machine Learning Algorithms)

### Variables for the model Cumple (Traget Variable) and Ingreso_Salida / Cluster_AGENCIA / Cluster_SITIOS as dependent variables.

datos_recaladas_limpio.columns
modelo_datos_recaladas = datos_recaladas_limpio[['Ingreso_Salida', 'Cluster_AGENCIA', 'Cluster_SITIOS', 'FECHA_ESTIMADA_DIA', 'FECHA_EFECTIVA_DIA', 'Cumple']]

# Convert categorical columns to numeric (label encoding)
modelo_datos_recaladas['Ingreso_Salida'] = modelo_datos_recaladas['Ingreso_Salida'].astype('category').cat.codes
modelo_datos_recaladas['Cluster_AGENCIA'] = modelo_datos_recaladas['Cluster_AGENCIA'].astype('category').cat.codes
modelo_datos_recaladas['Cluster_SITIOS'] = modelo_datos_recaladas['Cluster_SITIOS'].astype('category').cat.codes
modelo_datos_recaladas['Cumple'] = modelo_datos_recaladas['Cumple'].astype('category').cat.codes

# Drop date columns since we don't need them for modeling
modelo_datos_recaladas = modelo_datos_recaladas.drop(columns=['FECHA_ESTIMADA_DIA', 'FECHA_EFECTIVA_DIA'])

# Split the data into training (80%) and test (20%)
train_data, test_data, train_labels, test_labels = train_test_split(
    modelo_datos_recaladas.drop(columns=['Cumple']),
    modelo_datos_recaladas['Cumple'],
    test_size=0.2,
    random_state=42
)

print("Training data size:", len(train_data))
print("Testing data size:", len(test_data))

# 1. Decision Tree Model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(train_data, train_labels)

# Predicting on the test set
predictions_tree = decision_tree_model.predict(test_data)

# Decision Tree Model Evaluation
print("\nDecision Tree Model Evaluation:")
print(classification_report(test_labels, predictions_tree))
print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions_tree))

# Visualize the Decision Tree (Optional)
plt.figure(figsize=(15, 10))
plot_tree(decision_tree_model, filled=True, feature_names=train_data.columns)
plt.show()

# 2. Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=500, random_state=42)
random_forest_model.fit(train_data, train_labels)

# Predicting on the test set
predictions_rf = random_forest_model.predict(test_data)

# Random Forest Model Evaluation
print("\nRandom Forest Model Evaluation:")
print(classification_report(test_labels, predictions_rf))
print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions_rf))

# 3. XGBoost Model

# Prepare data for XGBoost
dtrain = xgb.DMatrix(train_data, label=train_labels)
dtest = xgb.DMatrix(test_data, label=test_labels)

# Define parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'max_depth': 6,
    'eta': 0.1
}

# Train the XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions with XGBoost
predictions_xgb = xgb_model.predict(dtest)
predictions_xgb = np.where(predictions_xgb > 0.5, 1, 0)

# XGBoost Model Evaluation
print("\nXGBoost Model Evaluation:")
print(classification_report(test_labels, predictions_xgb))
print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions_xgb))

# Feature importance for XGBoost
xgb.plot_importance(xgb_model)
plt.show()

# 4. Model Comparison

# Extract accuracy, sensitivity, specificity from the models
def extract_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    matrix = confusion_matrix(true_labels, predictions)
    sensitivity = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])  # Sensitivity (Recall for positive class)
    specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])  # Specificity (Recall for negative class)
    return accuracy, sensitivity, specificity

# Decision Tree Metrics
accuracy_tree, sensitivity_tree, specificity_tree = extract_metrics(test_labels, predictions_tree)

# Random Forest Metrics
accuracy_rf, sensitivity_rf, specificity_rf = extract_metrics(test_labels, predictions_rf)

# XGBoost Metrics
accuracy_xgb, sensitivity_xgb, specificity_xgb = extract_metrics(test_labels, predictions_xgb)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [accuracy_tree, accuracy_rf, accuracy_xgb],
    'Sensitivity': [sensitivity_tree, sensitivity_rf, sensitivity_xgb],
    'Specificity': [specificity_tree, specificity_rf, specificity_xgb]
})

# Round the results to 2 decimal places
comparison_df = comparison_df.round(2)

# Display the comparison
print("\nModel Comparison:")
print(comparison_df)





