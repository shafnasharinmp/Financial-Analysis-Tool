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
              missing_values = toString(months_seq[!months_seq %in% PERIOD_DATE]),
    )
  
  summ_list[[i]] <- summ_acc
  i = i+1
}

View(summ_list[[1]])

##########################################################################
summ_df <- data.frame(summ_list[[1]]) %>%
  filter(Zero_Count == 0 & Negative_Count == 0 & missing_count == 0) %>%
  reframe(ACCOUNT_ID,
          MARKET,
          CHANNEL_ID,
          MPG_ID,
          Frequency,
          Frequency_Range)
View(summ_df)

ts_obj <- ts(data = summ_df$MPG_ID, start = c(min(year(df$PERIOD_DATE)), 1), end = c(max(year(df$PERIOD_DATE)), 12), frequency = 12)
ts_obj
plot(ts_obj)

#####################################################################
k = 0

ts_data <- df %>%
  filter(df$ACCOUNT_ID == uniq_acc[1] & AMOUNT > 0) 
    
ts_data <- ts_data %>%
  group_by(ACCOUNT_ID,
             MARKET,
             CHANNEL_ID,
             MPG_ID) %>%
  filter(n() >= 45)

View(ts_data)

ts_obj1 <- ts(data = ts_data$AMOUNT, start = c(year(min(ts_data$PERIOD_DATE)), month(min(ts_data$PERIOD_DATE)), 1), 
              end = c(year(max(ts_data$PERIOD_DATE)), month(max(ts_data$PERIOD_DATE))), frequency = 12)

ts_obj1
plot(ts_obj1)

library(ggplot2)

decomp_ts <- decompose(ts_obj1, type = "multiplicative")

ggplot() +
  geom_line(aes(x = time(ts_obj1), y = ts_obj1, color = "Original")) +
  geom_line(aes(x = time(decomp_ts$trend), y = decomp_ts$trend, color = "Trend")) +
  labs(title = "Time Series with Trend Component",
       x = "Time",
       y = "Value") +
  scale_color_manual(values = c("Original" = "blue", "Trend" = "red")) +
  theme_minimal()

