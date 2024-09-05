
library(fpp3)
library(lubridate)
library(readxl)
library(ggplot2)
library(dbplyr)
library(forecast)
library(kableExtra)

# 1. Entendimiento General de los datos + limpieza de NAs y Outliers

datos_recaladas <- read_excel("Recaladas_Puerto_San_Antonio.xlsx")
glimpse(datos_recaladas)
summary(datos_recaladas)
colnames(datos_recaladas)

### Rango de fechas entre  01-01-2023 y 21-08-2024
min(datos_recaladas$`FECHA EFECTIVA`)
max(datos_recaladas$`FECHA EFECTIVA`)

### Atributos requeridos para el modelamiento
### Sí , FECHA EFECTIVA <= FECHA ESTIMADA, se asigna 1 a la columna Cumple, de lo contrario 0
### Adicionalmente, se pueden dar adelantos, en donde Fecha Efectiva < Fecha Estimada

datos_recaladas$FECHA_ESTIMADA_DIA <- as.Date(datos_recaladas$`FECHA ESTIMADA`, tryFormats = c("%Y-%m-%d"))
datos_recaladas$FECHA_EFECTIVA_DIA <- as.Date(datos_recaladas$`FECHA EFECTIVA`, tryFormats = c("%Y-%m-%d"))

datos_recaladas <- datos_recaladas %>% 
  mutate(Cumple = ifelse(FECHA_ESTIMADA_DIA >= FECHA_EFECTIVA_DIA, 1, 0 ))

datos_recaladas <- datos_recaladas %>% 
  mutate(Dias_Atraso = ifelse(Cumple == 1, 0, FECHA_EFECTIVA_DIA - FECHA_ESTIMADA_DIA ))

datos_recaladas <- datos_recaladas %>% 
  mutate(Adelanto = ifelse(FECHA_ESTIMADA_DIA > FECHA_EFECTIVA_DIA, 1, 0 ))

datos_recaladas <- datos_recaladas %>% 
  mutate(Dias_Adelanto = ifelse(Adelanto == 1, FECHA_ESTIMADA_DIA - FECHA_EFECTIVA_DIA,0 ))

datos_recaladas <- datos_recaladas %>% 
  mutate(Diferencia = FECHA_EFECTIVA_DIA - FECHA_ESTIMADA_DIA)

### Manejo de NA
sum(is.na(datos_recaladas)) # 227 NA en la data
sapply(datos_recaladas, function(x) sum(is.na(x))) #Filas sin SITIO de recalada
datos_recaladas_limpio <- datos_recaladas %>% drop_na()

### Datos Diferencia de días respecto a fecha estimada - Existe una gran dispersión en la diferencia entre la Fecha Estimada y Fecha Efectiva. Se analizarán posibles Outliers
diferencial_dias <- datos_recaladas_limpio %>% 
  group_by(Diferencia) %>% 
  summarise(Total = n()) %>% 
  mutate(
    Porcentaje = round((Total / sum(Total)) * 100, 3),  # Calcular porcentaje con 2 decimales
    Porcentaje_Acumulado = round(cumsum(Porcentaje), 3) # Calcular porcentaje acumulado con 2 decimales
  )

print(diferencial_dias)

### Histograma Diferencia días respecto a fecha estimada
Gráfico1 <- ggplot(datos_recaladas_limpio, aes(x = Diferencia)) + 
  geom_histogram(binwidth = 1.0, fill = "blue", color = "black", alpha = 0.5) + 
  geom_vline(xintercept = upper_threshold, color = "grey", linetype = "dashed", size = 0.5) + 
  geom_vline(xintercept = lower_threshold, color = "grey", linetype = "dashed", size = 0.5) + 
  labs(
    title = "Histograma de Diferencia días Fecha Estimada vs Fecha Efectiva",
    x = "Diferencia (días) - [Negativo(Adelantos)/Positivo(Atrasos)]",
    y = "Frecuencia"
  ) + 
  theme_minimal()

### Identificación de casos atípicos - regla de IQR*1.5 / NO APLICA

iqr<-IQR(datos_recaladas_limpio$Diferencia)
first <- quantile(datos_recaladas_limpio$Diferencia,prob= 0.25)
third <- quantile(datos_recaladas_limpio$Diferencia,prob= 0.75)
upper_threshold<- third+1.5*iqr
lower_threshold<- first-1.5*iqr

### Se considera estadísticamente casos atípicos valores con Adelantos mayores a 3 días y Atrasos mayores a 5 días.
### Recaladas con Adelantos mayores a 3 días: TBD Casos
### REcaladas con Atrasos mayores a 5 días: TBD Casos

outlier_function <- function(x) {
  iqr <- IQR(x, na.rm = TRUE)
  first <- quantile(x, probs = 0.25, na.rm = TRUE)
  third <- quantile(x, probs = 0.75, na.rm = TRUE)
  upper_threshold <- third + 1.5 * iqr
  lower_threshold <- first - 1.5 * iqr
  upper_outliers <- x[x > upper_threshold]
  lower_outliers <- x[x < lower_threshold]
  upper_outlier_count <- length(upper_outliers)
  lower_outlier_count <- length(lower_outliers)
  
  outlier <- list(
    upper_outlier_count = upper_outlier_count,
    lower_outlier_count = lower_outlier_count,
    upper_outliers = upper_outliers,
    lower_outliers = lower_outliers,
    upper_threshold = upper_threshold,
    lower_threshold = lower_threshold
  )
  
  return(outlier)
}

resultados_outliers <- outlier_function(datos_recaladas_limpio$Diferencia)

### Base de dato excluyendo Casos atípicos

# datos_recaladas_limpio <- datos_recaladas_limpio %>% 
  # filter(Diferencia < 5 & Diferencia > -3)

### Histograma base de datos sin NA y Casos Atípicos
Gráfico2 <- ggplot(datos_recaladas_limpio, aes(x = Diferencia)) + 
  geom_histogram(binwidth = 1.0, fill = "blue", color = "black", alpha = 0.5) + 
  labs(
    title = "Histograma de Diferencia días Fecha Estimada vs Fecha Efectiva",
    x = "Diferencia (días) - [Negativo(Adelantos)/Positivo(Atrasos)]",
    y = "Frecuencia"
  ) + 
  theme_minimal()

table(datos_recaladas_limpio$Diferencia)

# 2. Generación de nuevos atributos y cruce con base de dato meteorológica del día Estiado (Velocidad y dirección Viento)

### Generar un cluster para las AGENCIAS
datos_recaladas_limpio <- datos_recaladas_limpio %>%
  mutate(Cluster_AGENCIA = case_when(
    AGENCIA %in% c("AGENCIAS MARITIMAS AGENTAL LIMITADA", "AGENTAL") ~ "AGENTAL",
    AGENCIA == "AGENCIAS UNIVERSALES S.A." ~ "AGENCIAS UNIVERSALES",
    AGENCIA %in% c("B & M AGENCIA MARITIMA S.A.", "B&M") ~ "B & M AGENCIA MARITIMA",
    AGENCIA == "AGUNSA" ~ "AGUNSA",
    AGENCIA == "BROOM" ~ "BROOM",
    AGENCIA %in% c("EVERGREEN", "EVERGREEN SHIPPING AGENCY (CHILE)","EVERGREEN SHIPPING AGENCY ( CHILE )") ~ "EVERGREEN",
    AGENCIA == "IAN TAYLOR Y CIA. S.A." ~ "IAN TAYLOR",
    AGENCIA %in% c("INCHCAPE", "INCHCAPESHIPPING SERVICES B.V. CHILE LIMITADA") ~ "INCHCAPE",
    AGENCIA == "MARVAL" ~ "MARVAL",
    AGENCIA %in% c("MEDITERRANEAN SHIPPING COMPANY (CHILE) S.A.", "MSC") ~ "MSC",
    AGENCIA == "MTA" ~ "MTA",
    AGENCIA %in% c("SOCIEDAD MARITIMA Y COMERCIAL SOMARCO LTDA.", "SOMARCO") ~ "SOMARCO",
    AGENCIA == "TAYLOR" ~ "IAN TAYLOR",
    AGENCIA %in% c("ULTRAMAR", "ULTRAMAR AGENCIA MARITIMA LIMITADA") ~ "ULTRAMAR",
    TRUE ~ "Otros"  # Para cualquier agencia que no coincida con los anteriores
  ))

### Validación de Cluster Agencias
Cluster_Agencias <- datos_recaladas_limpio %>% 
  group_by(AGENCIA, Cluster_AGENCIA) %>% 
  summarise(servicios = n()) %>% 
  arrange(desc(servicios))

### Cumplimiento de la fecha estimada según AGENCIA
Cumplimiento_AGENCIAS <- datos_recaladas_limpio %>% 
  group_by(Cluster_AGENCIA) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(desc(Porcentaje_Cumplimiento))

### El cumplimiento varía de manera considerable dependiendo de la AGENCIA (Considerar en el modelo)

### Tabla resumen Cumplimiento_Agencias
Cumplimiento_AGENCIAS %>%
  mutate(across(everything(), ~ cell_spec(., align = "center"))) %>%
  kable("html", escape = FALSE, table.attr = "style='width:50%; margin-left:auto; margin-right:auto;'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  row_spec(0, align = "center")  # Center header


### Generar un cluster para las Sitios
datos_recaladas_limpio <- datos_recaladas_limpio %>%
  mutate(Cluster_SITIOS = case_when(
    grepl("PUERTO DE SAN ANTONIO", SITIO) ~ "Puerto de San Antonio",
    grepl("TERMINAL COSTANERA", SITIO) ~ "Terminal Costanera",
    SITIO == "DP World" ~ "DP World",
    grepl("STI", SITIO) & !grepl("EMPRESA PORTUARIA SAN ANTONIO|TERMINAL PORTUARIO PUERTO PANUL", SITIO) ~ "STI (Terminal Internacional)",
    SITIO == "QC" ~ "QC",
    grepl("EMPRESA PORTUARIA SAN ANTONIO \\(EPSA\\)", SITIO) ~ "EPSA",  
    grepl("VOPAK TERMINAL SAN ANTONIO LTDA", SITIO) ~ "VOPAK",
    grepl("TERMINAL PORTUARIO PUERTO PANUL", SITIO) ~ "PANUL",
    TRUE ~ "Otros"  # Para cualquier sitio que no coincida con los anteriores
  ))

### Validación de Cluster Sitios
Cluster_SITIO <- datos_recaladas_limpio %>% 
  group_by(SITIO, Cluster_SITIOS) %>% 
  summarise(servicios = n()) %>% 
  arrange(desc(servicios))

### Cumplimiento de la fecha estimada según SITIO
Cumplimiento_SITIO <- datos_recaladas_limpio %>% 
  group_by(Cluster_SITIOS) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(desc(Porcentaje_Cumplimiento))

### El cumplimiento varía de manera considerable dependiendo de la SITIO de Recalada (Considerar en el modelo)

### Tabla resumen Cumplimiento_Sitio
Cumplimiento_SITIO %>%
  mutate(across(everything(), ~ cell_spec(., align = "center"))) %>%
  kable("html", escape = FALSE, table.attr = "style='width:50%; margin-left:auto; margin-right:auto;'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  row_spec(0, align = "center")  # Center header


### Atributos de Fecha - Mes / Día_Mes / Día_semana / Periodo_Mes

datos_recaladas_limpio$mes_estimado <- month(datos_recaladas_limpio$`FECHA ESTIMADA`)
datos_recaladas_limpio$día_estimado <- day(datos_recaladas_limpio$`FECHA ESTIMADA`)
datos_recaladas_limpio$weekday <- wday(datos_recaladas_limpio$`FECHA ESTIMADA`)

datos_recaladas_limpio <- datos_recaladas_limpio %>%
  mutate(Cluster_Día = case_when(
    día_estimado >= 1 & día_estimado <= 10 ~ "Inicio Mes",
    día_estimado >= 11 & día_estimado <= 20 ~ "Medio Mes",
    día_estimado >= 21 & día_estimado <= 31 ~ "Final Mes",
    TRUE ~ "Desconocido"  # Esto es solo una medida de seguridad en caso de días fuera del rango
  ))

### Sumar Viento y Lluvias al modelo

datos_recaladas_limpio <- datos_recaladas_limpio %>% 
  left_join(datos_meteorológicos_final, by = c("FECHA_ESTIMADA_DIA" = "date"))

names(datos_recaladas_limpio)[names(datos_recaladas_limpio) == 'INGRESO O SALIDA'] <- 'Ingreso_Salida'

# 3. Análisis de correlación de variables independientes (Numpéricas) con el Atributo Cumple nivel de servicio

glimpse(datos_recaladas_limpio)

corr_df <- datos_recaladas_limpio %>% 
  select(Ingreso_Salida,Cluster_AGENCIA,Cluster_SITIOS,FECHA_ESTIMADA_DIA,FECHA_EFECTIVA_DIA,mes_estimado,día_estimado,weekday,tavg,prcp,wdir,wspd,Lluvia,Viento_Fuerte,Cumple)

glimpse(corr_df)

### Análisis de Correlación con variables Contínuas , no Categóricas

correlation <- cor(corr_df[,c(6,7,8,9,10,11,12,13,14,15)])
correlation

library(corrplot)

corrplot(correlation, method = "circle")
corrplot(correlation, method = "number")

# No se ven grandes correlaciones para incoporar estas variables al modelo.
# 4. Vamos a crear un modelo de predicción usando las variables: Ingreso_Salida / Cluster Agencia / Cluster Sitio - sin excluir los Outliers.

modelo_datos_recaladas <- corr_df %>% 
select(Ingreso_Salida, Cluster_AGENCIA, Cluster_SITIOS, FECHA_ESTIMADA_DIA, FECHA_EFECTIVA_DIA,Cumple)

glimpse(modelo_datos_recaladas)

### Separar nuestra data en entrenamiento (80%) y prueba (20%)

### install.packages("caret")
### install.packages("rpart")
### install.packages("randomForest")
### install.packages("rpart.plot")
### install.packages("rattle")

library(caret)
library(rpart)
library(randomForest)
library(rpart.plot)
library(rattle)

set.seed(42)

glimpse(modelo_datos_recaladas)

### Transformar Atributos a Factores
modelo_datos_recaladas$Ingreso_Salida <- as.factor(modelo_datos_recaladas$Ingreso_Salida)
modelo_datos_recaladas$Cluster_AGENCIA <- as.factor(modelo_datos_recaladas$Cluster_AGENCIA)
modelo_datos_recaladas$Cluster_SITIOS <- as.factor(modelo_datos_recaladas$Cluster_SITIOS)
modelo_datos_recaladas$Cumple <- as.factor(modelo_datos_recaladas$Cumple)  

trainIndex <- createDataPartition(modelo_datos_recaladas$Cumple, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

datos_train <- modelo_datos_recaladas[trainIndex, ]
datos_test <- modelo_datos_recaladas[-trainIndex, ]

cat("Tamaño del conjunto de entrenamiento:", nrow(datos_train), "\n")
cat("Tamaño del conjunto de prueba:", nrow(datos_test), "\n")

prop.table(table(datos_train$Cumple))
prop.table(table(datos_test$Cumple))

# 5. Modelo de Arbol de decisión (Decision Tree)

modelo_arbol <- rpart(Cumple ~ ., data = datos_train, method = "class")

printcp(modelo_arbol)
fancyRpartPlot(modelo_arbol)

predicciones <- predict(modelo_arbol, newdata = datos_test, type = "class")

matriz_confusion <- confusionMatrix(predicciones, datos_test$Cumple)
print(matriz_confusion)

# 5. Modelo de datos Random Forest 

modelo_rf <- randomForest(Cumple ~ ., data = datos_train, ntree = 500, mtry = 3, importance = TRUE)
print(modelo_rf)

predicciones_rf <- predict(modelo_rf, newdata = datos_test)

matriz_confusion_rf <- confusionMatrix(predicciones_rf, datos_test$Cumple)

print(matriz_confusion_rf)

# 6. Modelo de datos XGBoost

# Cargar la librería
library(xgboost)
library(caret)

# Convertir variables categóricas a numéricas
datos_preparados <- modelo_datos_recaladas
datos_preparados$Ingreso_Salida <- as.numeric(as.factor(datos_preparados$Ingreso_Salida))
datos_preparados$Cluster_AGENCIA <- as.numeric(as.factor(datos_preparados$Cluster_AGENCIA))
datos_preparados$Cluster_SITIOS <- as.numeric(as.factor(datos_preparados$Cluster_SITIOS))

# Convertir Cumple de 1, 2 a 0, 1
datos_preparados$Cumple <- as.numeric(as.factor(datos_preparados$Cumple)) - 1

# Remover columnas de fechas (FECHA_ESTIMADA_DIA y FECHA_EFECTIVA_DIA)
datos_preparados <- datos_preparados[, !(colnames(datos_preparados) %in% c("FECHA_ESTIMADA_DIA", "FECHA_EFECTIVA_DIA"))]

# Dividir los datos en entrenamiento y prueba
set.seed(123)
trainIndex <- createDataPartition(datos_preparados$Cumple, p = 0.7, list = FALSE)
datos_train_xg <- datos_preparados[trainIndex, ]
datos_test_xg <- datos_preparados[-trainIndex, ]

# Convertir los datos en matrices para XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(datos_train_xg[, -which(names(datos_train_xg) == "Cumple")]), label = datos_train_xg$Cumple)
test_matrix <- xgb.DMatrix(data = as.matrix(datos_test_xg[, -which(names(datos_test_xg) == "Cumple")]), label = datos_test_xg$Cumple)

# Definir los parámetros del modelo
params <- list(
  objective = "binary:logistic", # Clasificación binaria
  eval_metric = "error",         # Métrica de evaluación: tasa de error
  max_depth = 6,                 # Profundidad máxima del árbol
  eta = 0.1,                     # Tasa de aprendizaje
  nthread = 2                    # Número de hilos de procesamiento
)

# Entrenar el modelo
modelo_xgb <- xgboost(
  params = params,
  data = train_matrix,
  nrounds = 100, # número de iteraciones
  verbose = 1
)

# Predecir con el modelo entrenado
predicciones <- predict(modelo_xgb, test_matrix)
predicciones_binarias <- ifelse(predicciones > 0.5, 1, 0)

matriz_confusion_xg <- confusionMatrix(as.factor(predicciones_binarias), as.factor(datos_test_xg$Cumple))

importancia <- xgb.importance(feature_names = colnames(datos_train_xg[, -which(names(datos_train_xg) == "Cumple")]), model = modelo_xgb)
print(importancia)

xgb.plot.importance(importance_matrix = importancia)

# 7. Comparación de modelos y resultados finales

# Extraer métricas del modelo Árbol de Decisión
accuracy_arbol <- matriz_confusion$overall['Accuracy']
sensitivity_arbol <- matriz_confusion$byClass['Sensitivity']
specificity_arbol <- matriz_confusion$byClass['Specificity']
kappa_arbol <- matriz_confusion$overall['Kappa']
balanced_accuracy_arbol <- matriz_confusion$byClass['Balanced Accuracy']

# Extraer métricas del modelo Random Forest
accuracy_rf <- matriz_confusion_rf$overall['Accuracy']
sensitivity_rf <- matriz_confusion_rf$byClass['Sensitivity']
specificity_rf <- matriz_confusion_rf$byClass['Specificity']
kappa_rf <- matriz_confusion_rf$overall['Kappa']
balanced_accuracy_rf <- matriz_confusion_rf$byClass['Balanced Accuracy']

# Extraer métricas del modelo XG Boost
accuracy_xg <- matriz_confusion_xg$overall['Accuracy']
sensitivity_xg <- matriz_confusion_xg$byClass['Sensitivity']
specificity_xg <- matriz_confusion_xg$byClass['Specificity']
kappa_xg <- matriz_confusion_xg$overall['Kappa']
balanced_accuracy_xg <- matriz_confusion_xg$byClass['Balanced Accuracy']

# Crear un data frame de resumen comparativo
resumen_comparativo <- data.frame(
  Modelo = c('Árbol de Decisión', 'Random Forest', 'Modelo XG'),
  Accuracy = c(accuracy_arbol, accuracy_rf, accuracy_xg),
  Sensitivity = c(sensitivity_arbol, sensitivity_rf, sensitivity_xg),
  Specificity = c(specificity_arbol, specificity_rf, specificity_xg),
  Kappa = c(kappa_arbol, kappa_rf, kappa_xg),
  BalancedAccuracy = c(balanced_accuracy_arbol, balanced_accuracy_rf, balanced_accuracy_xg)
)

resumen_comparativo <- resumen_comparativo %>%
  mutate(across(where(is.numeric), ~ round(., 2)))
