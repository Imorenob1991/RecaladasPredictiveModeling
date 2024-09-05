
# https://meteostat.net/en/place/cl/san-antonio?s=85586&t=2023-01-01/2024-08-22#google_vignette

library(fpp3)
library(lubridate)
library(readr)
library(ggplot2)
library(dbplyr)
library(forecast)
library(kableExtra)

datos_meteorológicos <- read_csv("export.csv")

glimpse(datos_meteorológicos)
summary(datos_meteorológicos)

datos_meteorológicos_final <- datos_meteorológicos %>% 
  select(date,tavg, prcp, wdir, wspd)

datos_meteorológicos_final <- datos_meteorológicos_final %>% 
  mutate(Lluvia = ifelse(prcp > 0, 1, 0 ))

datos_meteorológicos_final <- datos_meteorológicos_final %>% 
  mutate(Viento_Fuerte = ifelse(wspd >= 20, 1, 0 ))
