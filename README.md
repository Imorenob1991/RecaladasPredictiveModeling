# RecaladasPredictiveModeling (Spanish)
Este proyecto analiza los datos de recaladas en el Puerto de San Antonio. Se han utilizado técnicas de modelamiento predictivo y análisis estadístico para evaluar el cumplimiento de las fechas estimadas de ingreso y salida. Los modelos predictivos analizados incluyen árboles de decisión, Random Forest y XGBoost. Los modelos fueron entrenados con el 80% de los datos disponibles y posteriormente evaluados en cuanto a su precisión con el 20% restante de la base de datos.

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
This project analyzes docking data at the Port of San Antonio(Chile), using predictive modeling techniques and statistical analysis to evaluate compliance with estimated entry and exit dates. It includes decision tree models, Random Forest, and XGBoost model to improve accuracy in predicting service levels.

# Descriptive analysis and data wrangling 
- Date range of data between 2023-01-01 and 2024-08-21

- Service level rule: If the Estimated Date equals the Actual Date, a value of 1 is assigned to the "Cumple(compliance)" column; otherwise, 0 is assigned.

- Out of the total 2,988 services, only 36.2% met the Estimated Date.

- There is a noticeable difference in compliance between Entries (41.2%) and Exits (31.1%).

- Compliance was then analyzed based on the Agency and Docking Site. Before this, it was necessary to create data maintainers to standardize the data (Example: ("AGENCIAS MARITIMAS AGENTAL LIMITADA", "AGENTAL") ~ "AGENTAL").

- A considerable difference in compliance is observed depending on the Agency and Docking Site:

<img width="562" alt="Cumplimiento_Agencia2" src="https://github.com/user-attachments/assets/e40717d1-a508-4d73-b374-586247d33de6">

<img width="560" alt="Cumplimiento_Sitio" src="https://github.com/user-attachments/assets/9465083a-2e6d-40e6-b898-9749f361aec7">

- Other date-related variables were analyzed, examining compliance by month, day of the month, day of the week, and period of the month (beginning, middle, end). The only variability detected was based on the day of the week, so this variable will be included in the predictive model.

- Finally, a predictive model will be developed based on the following variables: Entry or Exit, Cluster_AGENCY, Cluster_SITE, and weekday.

# Modelo Predictivo Cumplimiento de Fechas Estimadas







