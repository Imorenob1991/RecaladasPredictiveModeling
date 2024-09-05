# RecaladasPredictiveModeling (Spanish)
Este proyecto analiza los datos de recaladas en el Puerto de San Antonio. Se han utilizado técnicas de modelamiento predictivo y análisis estadístico para evaluar el cumplimiento de las fechas estimadas de ingreso y salida. Modelos de Machine Learning utilizados: árboles de decisión, Random Forest y XGBoost. Los modelos fueron entrenados con el 80% de los datos disponibles y posteriormente evaluados en cuanto a su precisión con el 20% restante de la base de datos.

# Análisis Descriptivo y Preparación de los Datos:
- Rango de fechas: Datos comprendidos entre el 01-01-2023 y el 21-08-2024.
- Cumplimiento del Nivel de Servicio: Se considera que se cumple cuando la fecha efectiva es menor o igual a la fecha estimada.
- Adelantos: Ocurren cuando la fecha efectiva es anterior a la fecha estimada. Es importante analizarlos, ya que también afectan la planificación.
- Limpieza de NAs: Se detectaron 227 servicios sin información de sitio de recalada.
- Outliers: Un servicio con un adelanto mayor a 3 días o un retraso mayor a 5 días se considera un dato atípico. Se evaluará si su exclusión mejora los resultados del modelo. En el gráfico siguiente, el valor 0 representa el cumplimiento exacto en el día estimado.

  <img width="607" alt="Gráfico_Histograma_Atraoss:Adelantos" src="https://github.com/user-attachments/assets/63c91ae7-c32d-4a33-b15c-a25108b7ea44">

- Análisis por Agencia y Sitio de Recalada: Se analizaron los cumplimientos según la agencia y el sitio de recalada. Previamente, fue necesario normalizar los datos para homogeneizarlos (por ejemplo, "AGENCIAS MARITIMAS AGENTAL LIMITADA" se normaliza a "AGENTAL").
- Diferencias Significativas: Se observó una diferencia considerable en el cumplimiento dependiendo de la agencia y el sitio:

  <img width="560" alt="Screenshot 2024-09-05 at 22 20 52" src="https://github.com/user-attachments/assets/bc3ac583-ea50-42ab-a56c-b8ccdc254532"><img width="564" alt="Screenshot 2024-09-05 at 22 21 37" src="https://github.com/user-attachments/assets/4c800142-9397-490d-bf71-b3a1e26812cf">

## Correlación con Variables Temporales y Meteorológicas:
- Se analizaron diferentes variables para entender si existía una correlación con el cumplimiento del nivel de servicio. El análisis no arrojó resultados significativos, ya que no se detectaron correlaciones relevantes con el cumplimiento: 

  <img width="720" alt="Screenshot 2024-09-05 at 22 29 53" src="https://github.com/user-attachments/assets/dddb14ae-448b-4fcd-8034-9a00537545cb">

# Modelo Predictivo Cumplimiento de Fechas Estimadas

El modelo predictivo se basa en las siguientes variables: Ingreso o Salida, Cluster_AGENCIA y Cluster_SITIO.

División de Datos: Los datos se separaron en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%).

Modelos Evaluados:

- Árbol de Decisión
- Random Forest
- XGBoost

  <img width="621" alt="Screenshot 2024-09-05 at 22 43 06" src="https://github.com/user-attachments/assets/6349144d-5498-4577-8df2-b1bcb4713a63">

- Mejor Desempeño: El modelo Random Forest mostró el mejor rendimiento, logrando predecir correctamente el 66% de los casos en la base de datos de prueba.

- Importancia de Variables: En el modelo XGBoost, la variable más relevante fue Cluster_AGENCIA (45%), seguida por Cluster_SITIO (44%) y, finalmente, Ingresos_Salida (11%).

Conclusiónes:
1. Con las variables tipo de servicio (Ingreso/Salida), agencia y sitio de recalada, se logró elaborar un modelo que predice correctamente el 66% de los casos, aunque aún hay espacio para mejoras.
2. La hipótesis inicial fue que las condiciones climáticas, como la velocidad del viento, su dirección y las precipitaciones, influirían en el cumplimiento del nivel de servicio.
3. Sin embargo, no se pudo demostrar esta hipótesis de manera estadística.
4. Próximos Pasos: Incorporar variables adicionales que puedan mejorar la precisión del modelo.

___________________________________________________________________________________________________________________________________________________________________________________

# RecaladasPredictiveModeling (English)
This project analyzes the berthing data at the Port of San Antonio. Predictive modeling techniques and statistical analysis have been used to evaluate compliance with estimated entry and exit dates. Machine Learning models used include decision trees, Random Forest, and XGBoost. The models were trained with 80% of the available data and subsequently evaluated for accuracy using the remaining 20% of the dataset.

# Descriptive Analysis and Data Wrangling
- Date Range: Data covers the period from 01-01-2023 to 21-08-2024.
- Service Level Compliance: Compliance is considered achieved when the actual date is less than or equal to the estimated date.
- Early Arrivals: Occur when the actual date is earlier than the estimated date. It is important to analyze them as they also affect planning.
- NA Cleaning: 227 services were found to have missing information on the berthing site.
- Outliers: A service with an early arrival of more than 3 days or a delay of more than 5 days is considered an outlier. It will be evaluated whether their exclusion improves the model results. In the following graph, the value 0 - represents exact compliance on the estimated day.

  <img width="607" alt="Gráfico_Histograma_Atraoss:Adelantos" src="https://github.com/user-attachments/assets/63c91ae7-c32d-4a33-b15c-a25108b7ea44">

- Analysis by Agency and Berthing Site: Compliance was analyzed according to the agency and berthing site. Previously, it was necessary to normalize the data for consistency (for example, "AGENCIAS MARITIMAS AGENTAL LIMITADA" was standardized to "AGENTAL").
- Significant Differences: A considerable difference in compliance was observed depending on the agency and the berthing site.

  <img width="560" alt="Screenshot 2024-09-05 at 22 20 52" src="https://github.com/user-attachments/assets/bc3ac583-ea50-42ab-a56c-b8ccdc254532"><img width="564" alt="Screenshot 2024-09-05 at 22 21 37" src="https://github.com/user-attachments/assets/4c800142-9397-490d-bf71-b3a1e26812cf">

## Correlation with Temporal and Meteorological Variables:
- Different variables were analyzed to determine if there was any correlation with service level compliance. The analysis did not yield significant results, as no relevant correlations with compliance were detected.

  <img width="720" alt="Screenshot 2024-09-05 at 22 29 53" src="https://github.com/user-attachments/assets/dddb14ae-448b-4fcd-8034-9a00537545cb">

# Predictive Model for Estimated On-Time Service Level

The predictive model is based on the following variables: Entry or Exit, Cluster_AGENCY, and Cluster_SITE.

Data Split: The data was divided into a training set (80%) and a test set (20%).

Evaluated Models:

- Decision Tree
- Random Forest
- XGBoost

  <img width="621" alt="Screenshot 2024-09-05 at 22 43 06" src="https://github.com/user-attachments/assets/6349144d-5498-4577-8df2-b1bcb4713a63">

- Best Performance: The Random Forest model showed the best performance, correctly predicting 66% of the cases in the test dataset.

- Variable Importance: In the XGBoost model, the most important variable was Cluster_AGENCY (45%), followed by Cluster_SITE (44%), and finally, Entry_Exit (11%).

Conclusions:

1. Using the variables service type (Entry/Exit), agency, and berthing site, a model was developed that correctly predicts 66% of the cases, although there is still room for improvement.
2. The initial hypothesis was that weather conditions, such as wind speed, direction, and precipitation, would influence service level compliance.
3. However, this hypothesis could not be statistically demonstrated.
4. Next Steps: Incorporate additional variables that could improve the model's accuracy.






