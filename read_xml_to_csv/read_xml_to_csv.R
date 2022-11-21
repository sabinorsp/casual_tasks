library('xml2')
library('purrr')
library('tidyverse')
library('plyr')

doc <- read_xml('Laptop_Train_v2.xml')

# Capturando as sentence que possui as tag <aspectTerms>
odr1 <- function(x){
  set_names( c(
    xml_find_all(x, './ancestor::sentence/text') %>% xml_text(),
    x %>% xml_attr('term'),
    x %>% xml_attr('polarity'),
    x %>% xml_attr('from'),
    x %>% xml_attr('to')
  ), c('text', 'term', 'polarity', 'from', 'to')) %>% as.list() %>% flatten_df()
}
  
df1 <- 
  xml_find_all(doc,".//aspectTerm") %>% 
  map_df(odr1) %>% type_convert()


# Caputando as sentence que não contém a presença da Tag <aspectTerms>
odr2 <- function(x){
  if( length(xml_find_all(x, './/aspectTerms')) == 0){
    set_names( c(
      xml_find_all(x, './/text') %>% xml_text()
    ), c('text')) %>% as.list() %>% flatten_df()
  }
}

df2 <- 
  xml_find_all(doc,".//sentence") %>%
  map_df(odr2) %>% type_convert()

# Fazendo a união dos dois dataframe df1 + df2 <- df
df <- rbind.fill(df1, df2)

#Permanece dados organizados em csv: 
write_csv(df, file='xml_organizado.csv')





