# NBA Shot Selection Optimization

## Overview
This project develops a ML model to optimize shot selection for NBA players by predicting shot success probabilities based on shot type, location, and game context. Using data from the NBA Stats API, the project includes exploratory data analysis (EDA), a Random Forest (RF) classification model, A/B testing to evaluate model recommendations, and an interactive Shiny dashboard for visualization of model findings. The goal is to provide actionable insights for coaches and analysts to improve team scoring efficiency.

## Features 
* Data Source: NBA Stats API (nba_api) for shot chart data from the 2022-23 season (NY Knicks).
* EDA: visualizations of shot success rates by zone, distance, and court location
* Classification Model: RF model to predict shot success probability using features such as: SHOT_TYPE, SHOT_ZONE_BASIC, SHOT_DISTANCE, LOC_X, LOC_Y, ACTION_TYPE, and PERIOD
* A/b Testing: comparing actual shot outcomes with model recommended shots (threshold probability > 0.5) to quantify improvements in expected points
* Interactive Dashboard: R Shiny app with heatmap of shot locations, success rates by zone, and expected points comparison

## Prerequisites
* Python 3.8+ with the following packages:
  ```pip install nba_api pandas numpy seaborn matplotlib scikit-learn scipy```
* R 4.0+ with the following packes:
  ```install.packages(c("shiny","ggplot2","dplyr", "plotly"))```
* Rstuio is what I use to run the shiny dashboard and recommend you do as well

## Running the Project
* Python analysis:
  * Run the script to fetch data, perform EDA, train the model, and conduct A/B testing:
    ``` python shot_selection_analysis.py```
  * Outputs
    * shot_data.csv which is the raw shot data from the NBA stats API
    * enhanced_shot_data.csv which is the processed data with model predictions
    * eda_plots folder which houses the visualizations
* Shiny dashboard
  * ensure enhanced_shot_data.csv is in the same directory as shot_selection_dashboard.R
  * open shot_selection_dashboard.R in Rstudio and click "Run App" or run in R:
    ```shiny::runApp("shot_selection_dashboard.R")```
  * the dashboard will open in your preferred browser, and will allow you to filter by player and adjust the shot probability threshold
 
# Future Roadmap 13 July
- [ ] improve a/b testing logic
- [ ] tuning RF model to achieve more accurate results and improvements on current 0.6 accuracy 
