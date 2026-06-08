# traffic-accident-risk-analysis
Traffic Accident Risk Prediction and Clustering Analysis using Python, K-Means, PCA and Random Forest

## Project Overview

This project analyzes 100,000 traffic accident records from the US Accidents dataset.

The main objective is to identify hidden accident patterns and predict accident risk levels using machine learning techniques.

## Dataset

US Accidents (March 2023 Release)

Sample Size: 100,000 records

## Project Workflow

### 1. Data Preprocessing

- Missing value analysis
- Feature selection
- Data cleaning
- Feature engineering

### 2. Exploratory Data Analysis

- Severity distribution analysis
- Accident frequency by hour
- Correlation analysis
- Weather condition grouping

### 3. Clustering Analysis

K-Means Clustering was applied to discover hidden accident profiles.

PCA was used to reduce dimensionality and visualize clusters.

### 4. Predictive Modeling

Random Forest Classification was used to predict accident risk levels.

Target Variable:

- Low Risk (Severity 1–2)
- High Risk (Severity 3–4)

## Results

### K-Means Clustering

- Identified 4 distinct accident profiles
- Visualized clusters using PCA

### Random Forest Classification

- Accuracy: 83%
- Identified key factors affecting accident severity

## Most Important Features

- Distance(mi)
- Start_Lat
- Start_Lng
- Pressure(in)
- Temperature(F)
- Wind Speed
- Hour of Day

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- K-Means
- PCA
- Random Forest

## Author

Öykü Küçük

Management Information Systems Student

Yeditepe University
