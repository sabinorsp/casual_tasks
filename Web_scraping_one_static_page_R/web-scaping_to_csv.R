# Code example for web-scraping

# Objective: 
# Extract the follow structure of the website : 'https://www.sportsmole.co.uk/football/premier-league/fixtures.html'
# Date, Time1, Time2
# save on output date_times.csv


# put off warnings
options(warn = -1)

# Load packages: 
library(xml2)
library(rvest)
library(readr)

# Read the webpage
link <- 'https://www.sportsmole.co.uk/football/premier-league/fixtures.html'
webpage <- read_html(link)
webpage

# Select the nodes by name class of html webpage and save on dataframe: 
date <- webpage %>% html_nodes('.game_state_0') %>% html_text(trim = T)
date
time1 <- webpage %>% html_nodes('.l_sfp_two') %>% html_text(trim = T)
time1
time2 <- webpage %>% html_nodes('.l_sfp_four') %>% html_text(trim = T)
time2

df <- data.frame(date = date, 
                 time1 = time1,
                 time2 = time2)
View(df)

#Drop the outlier values: 
df <- df[-(224:244), ]

# Export to a file .csv
write_csv(df, "times_datas.csv")
