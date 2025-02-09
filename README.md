# Air pollution Prediction using LSTM

## Introduction
This project is about predicting the air quality index using LSTM. The dataset used is bulk historical data of weather from OpenWeatherMap. The dataset contains the following columns:
- date 
- aqi (Air Quality Index)
- co (Carbon Monoxide)
- no (Nitrogen Monoxide)
- no2 (Nitrogen Dioxide)
- o3 (Ozone)
- so2 (Sulphur Dioxide)
- pm2_5 (Particulate Matter 2.5)
- pm10 (Particulate Matter 10)
- nh3 (Ammonia)

## Requirements
- Python 3.6
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Keras
- Tensorflow
- Jupyter Notebook
- OpenWeatherMap API
- Requests
- JSON
- Time
- Pickle
- Math
- Seaborn

## Steps
1. Data Collection
2. Data Preprocessing
3. Data Visualization
4. Model Building
5. Model Training
6. Model Evaluation
7. Model Testing
8. Conclusion


## Data Collection
The data is collected from OpenWeatherMap API. The data is collected for 5 years from 2020 to 2024. The data is stored in a CSV file.

## Data Preprocessing
The data is preprocessed by removing the null values and normalizing the data. The data is then split into training and testing data.

## Data Visualization
The data is visualized using Matplotlib and Seaborn. The data is visualized in the form of line plots, scatter plots, and histograms.

## Model Building
The model is built using LSTM. The model is built using Keras and Tensorflow.

## Model Training
The model is trained using the training data. The model is trained for 10 epochs.

## Model Evaluation
The model is evaluated using the testing data. The model is evaluated using the mean squared error.

## Model Testing
The model is tested using the testing data. The model is tested using the mean squared error.


## ML_Flow Integration
The project is integrated with MLFlow for tracking the model parameters and metrics. The model parameters and metrics are logged using MLFlow. Using grid search, the best model is selected.
The best model is then served using Flask.

## Live Data Prediction
The model is used to predict the air quality index using live data from OpenWeatherMap API. The model is used to predict the air quality index for the next 24 hours. The predicted data is then visualized using Matplotlib. 

## Grafana and Prometheus Integration
The project is integrated with Grafana and Prometheus for monitoring the model performance. The model performance is monitored using Grafana and Prometheus. The model performance is visualized using Grafana.

## Conclusion
The project is about predicting the air quality index using LSTM. The model is built using Keras and Tensorflow. The model is trained and tested using the data from OpenWeatherMap API. The model is then used to predict the air quality index for the next 24 hours. The model is integrated with MLFlow for tracking the model parameters and metrics. The model is also integrated with Grafana and Prometheus for monitoring the model performance. The model performance is visualized using Grafana. The project is successfully completed.