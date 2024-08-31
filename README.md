# RecaladasPredictiveModeling (Spanish)
Este proyecto analiza los datos de recaladas en el Puerto de San Antonio, utilizando técnicas de modelamiento predictivo y análisis estadístico para evaluar el cumplimiento de fechas estimadas de ingreso y salida. Incluye modelos de árboles de decisión, Random Forest y XGBoost para mejorar la precisión en la predicción del nivel de servicio.

# Análisis Descriptivo y Preparación de los Datos:
- Rango de fecha datos entre el 2023-01-01 al 2024-08-21

- Regla de cumplimiento nivel de servicio: Sí, Fecha Estimada = Fecha Efectiva, se asigna 1 a la columna cumple, de lo contrario 0.

- De los 2.988 servicios totales, solamente el 36.2% cumplió con la Fecha Estimada.

- Se observa una diferencia en el cumplimiento de Ingresos (41.2%) respecto a las salidas (31.1%)

- Luego se analizaron los cumplimientos en base a la Agencia y al Sitio de recalada. Previo, fue necesario generar mantenedores para homologar los datos (Ejemplo: ("AGENCIAS MARITIMAS AGENTAL LIMITADA", "AGENTAL") ~ "AGENTAL",)

- Se observa una diferencia considerable entre el cumplimiento dependiendo de la Agencia y Sitio:

  <img width="626" alt="Imagen_Agencia" src="https://github.com/user-attachments/assets/91984483-0826-4759-bb23-7a7f656516a7">

  <img width="629" alt="Imagen_Sitio" src="https://github.com/user-attachments/assets/7d98c61e-3d0b-49dd-89c2-95cad6d1aab9">

- Se analizan otras variables relacionadas a la fecha, analizando el cumplimiento por mes, día del mes, día de la semana y período del mes (inicio, mitad, fin). Sin embargo, no se encontraron diferencias significativas en estas variables que justifiquen su inclusión en el modelo predictivo.

- Finalmente se realizará un modelo predictivo en base a las variables: Ingreso o Salida, Cluster_AGENCIA y Cluster_SITIO

# Modelo Predictivo Cumplimiento de Fechas Estimadas

- Primero, se separan los datos entre la data de entrenamiento(80%) y prueba(20%)

- Se realizaron 3 modelos predictivos, de modelos más simples a más complejos:
  1. Arbol de Decisión.
  2. Random Forest
  3. XG Boost

<img width="680" alt="resumen_modelos_predictivos" src="https://github.com/user-attachments/assets/c2e52b64-829b-4a3f-befe-f21370428721">

- El modelo que tuvo mejor performance con las bases de datos de prueba fue el XG Boost, logrando predecir el 66% de los casos de manera correcta.

- Es importante considerar, que mediante la incoporación de más variables se podría enriquecer el modelo, aumentando su precisión.

- Del modelo actual, la variable con más relevancia en el modelo predictivo XG Boost es la variable Cluster_AGENCIA con un 59%, luego Cluster_SITIOS con un 32% y finalmente la variable Ingresos_Salida con un 9%

_________________________________________________________________________________________________________________________________

# RecaladasPredictiveModeling (English)
This project analyzes docking data at the Port of San Antonio, using predictive modeling techniques and statistical analysis to evaluate compliance with estimated entry and exit dates. It includes decision tree models, Random Forest, and XGBoost to improve accuracy in predicting service levels.

# Descriptive analysis and data wrangling 
- Date range of data between 2023-01-01 and 2024-08-21




