# casual_tasks
Este é um repositório que contém algumas tarefas casuais  que foram realizadas. 

---

## read_xml_to_csv: 
Script em R para conversão de um padrão xml para csv

---

## Web_scraping_one_static_page_R 
Web scrpaing de uma página estática, link: https://www.sportsmole.co.uk/football/premier-league/fixtures.html

Objetivo: Capturar os times que irão se enfrentar de acordo com a data do jogo e salvar numa estrutura .csv de acordo com a estruturra: 
data - Time1 - Time2

A script realiza a carga dos dados html e faz a busca dos nós através dos nomes das classes da variável desejada, por fim cria-se um dataframe com esses dados e permanece em arquivo .csv.

---

## extract_data_tradinfview: 

## Objective:
Extract data from TradingView using package Tv data feed:

Executing some rule to filter data;  
Saving the results on a sheet in google sheet.  
The API of TradingView are avaliable on: https://br.tradingview.com/rest-api-spec/  

## Instructions  
install package tv datafeed - pip install --upgrade --no-cache-dir git+https://github.com/StreamAlpha/tvdatafeed.git  
install gspread - pip install gspread  
install gsread_dataframe - pip install gspread_dataframe  
install pydrive - pip install pydrive  
install mplfinance - pip install mplfinance  
The code source about this package are present in: https://github.com/StreamAlpha/tvdatafeed.  
To some examples: https://github.com/StreamAlpha/tvdatafeed/blob/main/tv.ipynb.  

---
