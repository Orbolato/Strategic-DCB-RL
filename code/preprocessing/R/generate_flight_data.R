# Cleaning workspace
rm(list = ls())

# Libraries
library(dplyr)
library(lubridate)
library(readr)

# Setting work directory
folder <- "D:/Documentos/Doutorado Orbolato/Resultados/3 - Strategic Demand-Capacity Balancing/"
setwd(folder)

# Loading traffic data
load("data/sbgr_data.RData")
df$real_wind_state <- df$wind_state
df$real_weather_condition <- df$ifr + df$lifr
df <- subset(df, select = -c(vsby, 
                             ceiling,
                             ifr,
                             lifr,
                             drct,
                             sknt,
                             headwind_10,
                             crosswind_10,
                             wind_state))
print(names(df))

# 0) Ensure df$time is on a 15-min grid and create a stable slot_index
df_idx <- df %>%
  mutate(time = floor_date(time, unit = "15 minutes")) %>%
  arrange(time) %>%
  distinct(time, .keep_all = TRUE) %>%     # ensure one row per slot time
  mutate(slot_index = row_number())

# 1) Sort arrivals and compute slot_time (UTC)
siros_arr_sorted <- siros_arr %>%
  arrange(Chegada.Prevista_40) %>%
  mutate(slot_time = floor_date(Chegada.Prevista_40, unit = "15 minutes"))

# 2) Compute day_slot (0..95) for operational days starting at 06:00 UTC
#    day_start = floor_date(slot_time - 6h, "day") + 06:00
#    day_slot = minutes since day_start divided by 15
arrivals_joined <- siros_arr_sorted %>%
  mutate(
    op_day_start = floor_date(slot_time - hours(6), unit = "day") + hours(6),
    day_slot = as.integer(as.numeric(difftime(slot_time, op_day_start, units = "mins")) / 15)#,
    # ADD THIS: Extract the operational day as a date
    # day = as.Date(op_day_start)
  ) %>%
  # 3) Join to df to grab the matching slot row and its index
  left_join(df_idx, by = c("slot_time" = "time")) %>%
  select(-op_day_start)
  # Now you can keep op_day_start if needed, or just keep 'day'
arrivals_joined <- arrivals_joined[!is.na(arrivals_joined$scheduled_arrivals_40),]
arrivals_joined <- subset(arrivals_joined, select = -c(Operador.Aéreo,
                                                   Etapa,
                                                   Aeroporto.Origem,
                                                   Início,
                                                   Tipo,
                                                   Aeroporto.Destino,
                                                   tx_codeshare))

colnames(arrivals_joined)[1:10] <- c('icao',
                                   'flight_number',
                                   'equip',
                                   'seats',
                                   'origin',
                                   'scheduled_departure_time',
                                   'scheduled_arrival_time',
                                   'destination',
                                   'scheduled_arrival_40_time',
                                   'scheduled_arrival_100_time')

save(arrivals_joined, file="data/arrival_data_gru.Rda")
save.image("data/arrival_sbgr_data.RData")
write.csv(arrivals_joined, 'data/arrival_data.csv', row.names=FALSE)
