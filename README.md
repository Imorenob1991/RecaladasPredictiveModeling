# RecaladasPredictiveModeling (Spanish)
Este proyecto analiza los datos de recaladas en el Puerto de San Antonio. Se utilizaron técnicas de modelamiento predictivo y análisis estadístico para evaluar el cumplimiento de fechas estimadas de ingreso y salida. Se analizaron los siguientes modelos predictivos: árboles de decisión, Random Forest y XGBoost. Se entrenan los modelos con el 80% de la base de datos y luego se prueba su asertividad con el 20% de la base de datos restante.

# Análisis Descriptivo y Preparación de los Datos:
- Rango de fecha datos entre el 2023-01-01 al 2024-08-21
- Cumple Nivel de Servicio[1] : Fecha Efectiva <= Fecha Estimada.
- Adelantos: Fecha Efectiva < Fecha Estimada (Importante su análisis dado que afectan también a la planificación)
- Limpieza de NAs: Se observan 227 servicios sin información de Sitio de Recalada.
- Outliers: Se estima que un servicio con un adelanto mayor a 3 días o un retraso mayor a 5 días podría considerarse como un dato atípico. Sin embargo, se evaluará si su exclusión mejora el resultado del modelo. En el siguiente gráfico, el 0 representa el cumplimiento exacto en el día estimado:

  <img width="607" alt="Gráfico_Histograma_Atraoss:Adelantos" src="https://github.com/user-attachments/assets/63c91ae7-c32d-4a33-b15c-a25108b7ea44">

- Luego se analizaron los cumplimientos en base a la Agencia y al Sitio de recalada. Previo, fue necesario generar mantenedores para homologar los datos (Ejemplo: ("AGENCIAS MARITIMAS AGENTAL LIMITADA", "AGENTAL") ~ "AGENTAL",)
- Se observa una diferencia considerable entre el cumplimiento dependiendo de la Agencia y Sitio:

  <img width="560" alt="Screenshot 2024-09-05 at 22 20 52" src="https://github.com/user-attachments/assets/bc3ac583-ea50-42ab-a56c-b8ccdc254532"><img width="564" alt="Screenshot 2024-09-05 at 22 21 37" src="https://github.com/user-attachments/assets/4c800142-9397-490d-bf71-b3a1e26812cf">

## Correlación con Variables Temporales y Meteorológicas:
- Se analizaron distintar variables para entender si había una correlación entre ellas y el cunplimiento del nivel de servicio. El resultado no fue satisfactorio y no se identificaron correlaciones con el cumplimiento.

<img width="720" alt="Screenshot 2024-09-05 at 22 29 53" src="https://github.com/user-attachments/assets/dddb14ae-448b-4fcd-8034-9a00537545cb">

# Modelo Predictivo Cumplimiento de Fechas Estimadas

- Finalmente se realizará un modelo predictivo en base a las variables: Ingreso o Salida, Cluster_AGENCIA y Cluster_SITIO

- Primero, se separan los datos entre la data de entrenamiento(80%) y prueba(20%)

- Se realizaron 3 modelos predictivos, de modelos más simples a más complejos:
  1. Arbol de Decisión.
  2. Random Forest
  3. XG Boost

<img width="680" alt="resumen_modelos_predictivos" src="https://github.com/user-attachments/assets/c2e52b64-829b-4a3f-befe-f21370428721">

- El modelo que tuvo mejor performance con las bases de datos de prueba fue el XG Boost, logrando predecir el 66% de los casos de manera correcta.

- Es importante considerar, que mediante la incoporación de más variables se podría enriquecer el modelo, aumentando su precisión.

- Del modelo actual, la variable con más relevancia en el modelo predictivo XG Boost es la variable Cluster_AGENCIA con un 59%, luego Cluster_SITIOS con un 32% y finalmente la variable Ingresos_Salida con un 9%

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







