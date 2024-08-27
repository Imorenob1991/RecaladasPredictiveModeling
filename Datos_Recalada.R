
library(fpp3)
library(lubridate)
library(readxl)
library(ggplot2)
library(dbplyr)
library(forecast)

datos_recaladas <- read_excel("Recaladas_Puerto_San_Antonio.xlsx")
glimpse(datos_recaladas)
colnames(datos_recaladas)

### Fechas de la base de datos
min(datos_recaladas$`FECHA EFECTIVA`)
max(datos_recaladas$`FECHA EFECTIVA`)

# 1. Atributos necesarios para la realización del modelo, Cumplimiento Fecha Estimada
# 1.1 En caso de que Fecha Estimada = Fecha Efectiva, se considera que cumple el nivel de servicio. 

datos_recaladas$FECHA_ESTIMADA_DIA <- as.Date(datos_recaladas$`FECHA ESTIMADA`, tryFormats = c("%Y-%m-%d"))
datos_recaladas$FECHA_EFECTIVA_DIA <- as.Date(datos_recaladas$`FECHA EFECTIVA`, tryFormats = c("%Y-%m-%d"))

datos_recaladas <- datos_recaladas %>% 
  mutate(Cumple = ifelse(FECHA_ESTIMADA_DIA == FECHA_EFECTIVA_DIA, 1, 0 ))

# 2. Cumplimiento nivel de servicio

#2.1 Total - el 36.2% de los ingresos y salidas del puerto cumple la fecha estimada.

datos_recaladas %>% 
  group_by(Cumple) %>% 
  summarise(Total = n()) %>% 
  mutate( Porcentaje_Total = (Total/sum(Total)) * 100)

#2.2 Cumplimiento en Ingreos y Salidas - 41.2% de los ingresos cumple fecha estimada y el 31.1% de las salidas.

datos_recaladas %>% 
  filter(`INGRESO O SALIDA` == "INGRESO") %>% 
  group_by(Cumple) %>% 
  summarise(Total = n()) %>% 
  mutate( Porcentaje_Total = (Total/sum(Total)) * 100)            

datos_recaladas %>% 
  filter(`INGRESO O SALIDA` == "SALIDA") %>% 
  group_by(Cumple) %>% 
  summarise(Total = n()) %>% 
  mutate( Porcentaje_Total = (Total/sum(Total)) * 100)   

# 2.3 Cuimplimiento en base a Agencia Responsable ¿Existe alguna direrencia significativa?
# R: Sí, existe una diferencia relevante entre el cumplimiento y la Agencia a cargo del servicio.

### Generar un cluster para las AGENCIAS
datos_recaladas <- datos_recaladas %>%
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

### Validación de Cluster
Cluster_Agencias <- datos_recaladas %>% 
  group_by(AGENCIA, Cluster_AGENCIA) %>% 
  summarise(servicios = n()) %>% 
  arrange(desc(servicios))

### Cumplimiento de la fecha estimada según AGENCIA
Cumplimiento_AGENCIAS <- datos_recaladas %>% 
  group_by(Cluster_AGENCIA) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(desc(Porcentaje_Cumplimiento))

# 2.4 Cuimplimiento en base a Puerto de Recalada (SITIO) ¿Existe alguna direrencia significativa?
# R: Existe una diferencia significativa en función del sitio de Recalada.

### Generar un cluster para las Sitios de Recaladas
datos_recaladas <- datos_recaladas %>%
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

### Validación de Cluster
Cluster_SITIO <- datos_recaladas %>% 
  group_by(SITIO, Cluster_SITIOS) %>% 
  summarise(servicios = n()) %>% 
  arrange(desc(servicios))

### Cumplimiento de la fecha estimada según AGENCIA
Cumplimiento_SITIO <- datos_recaladas %>% 
  group_by(Cluster_SITIOS) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(desc(Porcentaje_Cumplimiento))

# 2.4 ¿Existe una mayor o menor probabilidad de cumplimiento dependiendo del meso día estimado del servicio?
# R: No se reconoce una diferencia significativa entre el día o mes del servicio y su porcentaje de cumplimiento.

datos_recaladas$mes_estimado <- month(datos_recaladas$`FECHA ESTIMADA`)
datos_recaladas$día_estimado <- day(datos_recaladas$`FECHA ESTIMADA`)
datos_recaladas$weekday <- wday(datos_recaladas$`FECHA ESTIMADA`)

datos_recaladas <- datos_recaladas %>%
  mutate(Cluster_Día = case_when(
    día_estimado >= 1 & día_estimado <= 10 ~ "Inicio Mes",
    día_estimado >= 11 & día_estimado <= 20 ~ "Medio Mes",
    día_estimado >= 21 & día_estimado <= 31 ~ "Final Mes",
    TRUE ~ "Desconocido"  # Esto es solo una medida de seguridad en caso de días fuera del rango
  ))

### Variabilidad de cumplimiento según mes
Cumplimiento_mensual <- datos_recaladas %>% 
  group_by(mes_estimado) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(mes_estimado)

### Variabilidad de cumplimiento según día del mes
Cumplimiento_diario <- datos_recaladas %>% 
  group_by(día_estimado) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(día_estimado)

### Variabilidad de cumplimiento según día de la semana
Cumplimiento_wday <- datos_recaladas %>% 
  group_by(weekday) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2)) %>% 
  arrange(weekday)

### Variabilidad de cumplimiento según etapa del mes (Inicio/Mitad/Fin)
Cumplimiento_diario_cluster <- datos_recaladas %>% 
  group_by(Cluster_Día) %>% 
  summarise(Servicios_Totales = n(),
            Cumplimiento = sum(Cumple)) %>% 
  mutate(Porcentaje_Cumplimiento = round(Cumplimiento / Servicios_Totales * 100, 2))

# 3. Vamos a crear un modelo de predicción usando las variables: Ingreso o Salida / Cluster Agencia / Cluster Sitio.

modelo_datos_recaladas <- datos_recaladas %>% 
  select(`INGRESO O SALIDA`, Cluster_AGENCIA, Cluster_SITIOS, FECHA_ESTIMADA_DIA, FECHA_EFECTIVA_DIA, Cumple)

### write.csv(modelo_datos_recaladas,"modelo_datos_recaladas.csv")

glimpse(modelo_datos_recaladas)

# 4. Separar nuestra data en entrenamiento (80%) y prueba (20%)

# install.packages("caret")
# install.packages("rpart")
# install.packages("randomForest")
install.packages("rpart.plot")
install.packages("rattle")

library(caret)
library(rpart)
library(randomForest)
library(rpart.plot)
library(rattle)

set.seed(42)  # Fijar la semilla para reproducibilidad

glimpse(modelo_datos_recaladas)

modelo_datos_recaladas$`INGRESO O SALIDA` <- as.factor(modelo_datos_recaladas$`INGRESO O SALIDA`)
modelo_datos_recaladas$Cluster_AGENCIA <- as.factor(modelo_datos_recaladas$Cluster_AGENCIA)
modelo_datos_recaladas$Cluster_SITIOS <- as.factor(modelo_datos_recaladas$Cluster_SITIOS)
modelo_datos_recaladas$Cumple <- as.factor(modelo_datos_recaladas$Cumple)  # Variable objetivo

trainIndex <- createDataPartition(modelo_datos_recaladas$Cumple, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

datos_train <- modelo_datos_recaladas[trainIndex, ]
datos_test <- modelo_datos_recaladas[-trainIndex, ]

names(modelo_datos_recaladas)[names(modelo_datos_recaladas) == 'INGRESO O SALIDA'] <- 'Ingreso_Salida'
names(datos_train)[names(datos_train) == 'INGRESO O SALIDA'] <- 'Ingreso_Salida'
names(datos_test)[names(datos_test) == 'INGRESO O SALIDA'] <- 'Ingreso_Salida'

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
  Modelo = c('Árbol de Decisión', 'Random Forest','Modelo XG'),
  Accuracy = c(accuracy_arbol, accuracy_rf,accuracy_xg),
  Sensitivity = c(sensitivity_arbol, sensitivity_rf,sensitivity_xg),
  Specificity = c(specificity_arbol, specificity_rf,specificity_xg),
  Kappa = c(kappa_arbol, kappa_rf,kappa_xg),
  BalancedAccuracy = c(balanced_accuracy_arbol, balanced_accuracy_rf,balanced_accuracy_xg)
)

# 8. Probar el modelo con datos Ficticios

### Crear un archivo

# Establecer la semilla para reproducibilidad
set.seed(42)

# Definir el número de filas
num_rows <- 100

# Definir las categorías para las columnas categóricas
categorias_agencia <- c('INCHCAPE', 'MSC', 'IAN TAYLOR', 'B & M AGENCIA MARITIMA', 'ULTRAMAR')
categorias_sitios <- c('EPSA', 'STI (Terminal Int.)', 'Terminal Costanera', 'PANUL', 'DP World')

# Generar datos aleatorios
Ingreso_Salida <- sample(c('INGRESO', 'SALIDA'), size = num_rows, replace = TRUE)
Cluster_AGENCIA <- sample(categorias_agencia, size = num_rows, replace = TRUE)
Cluster_SITIOS <- sample(categorias_sitios, size = num_rows, replace = TRUE)
FECHA_ESTIMADA_DIA <- seq.Date(from = as.Date('2024-01-01'), by = 'day', length.out = num_rows)
FECHA_EFECTIVA_DIA <- FECHA_ESTIMADA_DIA + sample(0:5, size = num_rows, replace = TRUE)

# Crear un data frame sin la columna 'Cumple'
prueba_prediccion <- data.frame(
  Ingreso_Salida,
  Cluster_AGENCIA,
  Cluster_SITIOS,
  FECHA_ESTIMADA_DIA,
  FECHA_EFECTIVA_DIA
)

prueba_prediccion2 <- prueba_prediccion

# Convertir variables categóricas a numéricas
prueba_prediccion2$Ingreso_Salida <- as.numeric(as.factor(prueba_prediccion2$Ingreso_Salida))
prueba_prediccion2$Cluster_AGENCIA <- as.numeric(as.factor(prueba_prediccion2$Cluster_AGENCIA))
prueba_prediccion2$Cluster_SITIOS <- as.numeric(as.factor(prueba_prediccion2$Cluster_SITIOS))

prueba_prediccion2 <- prueba_prediccion2[, !(colnames(prueba_prediccion2) %in% c("FECHA_ESTIMADA_DIA", "FECHA_EFECTIVA_DIA"))]

pred_matrix <- xgb.DMatrix(data = as.matrix(prueba_prediccion2))

predicciones <- predict(modelo_xgb, pred_matrix)

predicciones_binarias <- ifelse(predicciones > 0.5, 1, 0)

print(predicciones_binarias)

df_predicciones <- data.frame(Predicciones = predicciones_binarias)

# Resultado Final, Anexar la prdicción al a base de datos de prueba_predicción.

df_resultante <- cbind(prueba_prediccion, df_predicciones)

write.csv(df_resultante,"df_resultante.csv")

