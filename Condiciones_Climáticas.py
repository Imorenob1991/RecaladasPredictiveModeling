
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:15:02 2024

@author: momos_macbook_pro
"""

# Import necessary libraries
import pandas as pd
import numpy as np

# Load the CSV file
datos_meteorologicos = pd.read_csv("export.csv")

# Glimpse the data (equivalent to glimpse() in R)
print(datos_meteorologicos.info())
print(datos_meteorologicos.describe())

# Select the relevant columns (date, tavg, prcp, wdir, wspd)
datos_meteorologicos_final = datos_meteorologicos[['date', 'tavg', 'prcp', 'wdir', 'wspd']]

print(datos_meteorologicos_final.info())

# Create a new column 'Lluvia' where prcp > 0 is marked as 1 (indicating rain)
datos_meteorologicos_final['Lluvia'] = np.where(datos_meteorologicos_final['prcp'] > 0, 1, 0)

# Create a new column 'Viento_Fuerte' where wspd >= 20 is marked as 1 (indicating strong wind)
datos_meteorologicos_final['Viento_Fuerte'] = np.where(datos_meteorologicos_final['wspd'] >= 20, 1, 0)

# Display the first few rows to check the result
print(datos_meteorologicos_final.info())

datos_meteorologicos_final['date'] = pd.to_datetime(datos_meteorologicos_final['date'], format='%Y-%m-%d')
