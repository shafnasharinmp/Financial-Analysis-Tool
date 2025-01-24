rm(list = ls())
data = read.csv("F:/Timeseries Forecasting/gm_input_AP.csv")
df = data.table(data)
df$PERIOD_DATE = as.Date(df$PERIOD_DATE, format = "%d-%m-%Y")

################################################

freq_range <- function(freq){
  if(freq<12){
    range_freq = 1
  }else if(freq>=12 & freq<24){
    range_freq = 2
  }else if(freq>=24 & freq<36){
    range_freq = 3
  }else{
    range_freq = 4
  }
  return(range_freq)
}

outliers <- function(amount){
  quart1 = quantile(amount,0.25)
  quart3 = quantile(amount, 0.75)
  iqr = quart3 - quart1
  outlier_count = sum(amount < (quart1 - 1.5 * iqr) | amount > (quart3 + 1.5 * iqr))
  return(outlier_count)
}

#########################################

uniq_acc <- unique(df$ACCOUNT_ID)
summ_list <- list()

i = 1
while (i <= length(uniq_acc)) {
  acc <- df %>%
    filter(df$ACCOUNT_ID == uniq_acc[i])
  
  months_seq <- seq(from = min(acc$PERIOD_DATE), to = max(acc$PERIOD_DATE), by = "month")
  
  acc <- acc %>%
    group_by(ACCOUNT_ID,
             MARKET,
             CHANNEL_ID,
             MPG_ID)
  
  summ_acc <- acc %>%
    group_by(ACCOUNT_ID,
             MARKET,
             CHANNEL_ID,
             MPG_ID) %>%
    summarize(Frequency = n(),
              Frequency_Range = freq_range(Frequency),
              Zero_Count = sum(AMOUNT == 0),
              Negative_Count = sum(AMOUNT < 0),
              outlier_count = outliers(AMOUNT),
              missing_count = length(months_seq) - Frequency,
              missing_values = toString(months_seq[!months_seq %in% PERIOD_DATE])
              )

  summ_list[[i]] <- summ_acc
  i = i+1
}


wb <- createWorkbook()
addWorksheet(wb, "AP_Summary")

column_names = data.frame("ACCOUNT_ID", "MARKET", "CHANNEL_ID", "PRODUCT_ID",
                          "Frequency", "Frequency_Range", "Zero_count", "Negative_count", "Outliers_count", "Missing_count", "Missing_values")

writeData(wb, sheet = "AP_Summary", x = column_names, startRow = 1, rowNames = FALSE, colNames = FALSE, withFilter = FALSE)
start_row =nrow(column_names) + 1

k = 0
for (k in 1:length(summ_list)) {
  writeData(wb, sheet = "AP_Summary", x = summ_list[[k]], startRow = start_row, rowNames = FALSE, colNames = FALSE, withFilter = FALSE)
  start_row = start_row + nrow(summ_list[[k]])
}

saveWorkbook(wb, "Summary_AP_2.xlsx")

