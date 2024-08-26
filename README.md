# Predictive Modeling for Walmart Sales Using Time Series Forecasting
source dataset : https://www.kaggle.com/datasets/mikhail1681/walmart-sales

## Overview
Walmart is one of the biggest retailers in the world, known for its low-cost, high volume and multi channel approach. Walmart's target market include middle to low income individuals, as well as people who live in rural areas.

Walmart sales data is usually used for predicting and forecasting sales, optimizing sales, and analyzing customer behavior. To achieve this, we can conduct a comprehensive analysis of previous sales data.

## Data Description

| Column         | Description                      |
|----------------|----------------------------------|
| Store          | Store identifier               |
| Date           | Date of the transaction                 |
| Weekly Sales   | Weekly sales               |
| Holiday Flag   | National holiday (1 for yes, 0 for no) |
| Temperature    | Temperature in Fahrenheit |
| Fuel_Price     | Fuel price per gallon       |
| CPI            | Consumer Price Index |
| Unemployment   | Unemployment rate             |

- Rows: 6435
- Columns: 8
- Null values: 0

### Time
The data from:

- 2010: 5 February - 31 December, 330 days
- 2011: 1 January -  31 December, 365 days
- 2012: 1 january - 26 October, 294 days


## Exploratory Data Analysis

### Year
![SalesByYears](images/SalesByYearsBar.png)

### Month
![Month_Sales](images/MonthSalesLineComparison.png)

#### Factor
![Holiday](images/SalesInHoliday.png)
![Temp](images/SalesInTemp.png)
![CPI](images/WeeklyCPILine.png)
![Unemployment](images/WeeklyUnemploymentLine.png)




### Weekly
![Weekly_Sales](images/WeeklyWeekly_SalesLine.png)

## Analysis Approach
