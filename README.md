# Airbnb Pricing Recommendation Project (Columbus, Ohio)

## 📌 Project Overview
This project builds a machine learning model to recommend optimal nightly prices for Airbnb listings in Columbus, Ohio.

The goal is to support hosts and property managers in setting better prices based on listing characteristics and market data.

---

## 🎯 Objectives
- Predict Airbnb listing prices using machine learning
- Compare multiple regression models
- Evaluate model performance using standard metrics
- Build a reproducible experiment tracking process

---

## 📊 Dataset
Source: Inside Airbnb (Columbus, Ohio)

Data used:
- Listings dataset (`listings.csv.gz`)
- Calendar dataset (`calendar.csv.gz`) *(optional for time-based features)*

---

## ⚙️ Models
We will compare multiple regression models:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting / XGBoost (if implemented)

---

## 📈 Evaluation Metrics

### Technical Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)

### Business Metrics
- Revenue per available night
- Occupancy rate
- Pricing efficiency
- Time savings (conceptual)

---

## 🧪 Experiment Tracking
We track each experiment run with:
- Model type
- Hyperparameters
- Dataset version
- Features used
- Evaluation metrics (MAE, RMSE, R²)
- Notes and observations

---

## 📁 Project Structure


---

## 🧠 Key Concept
This project focuses on building a **price recommendation system**, not a perfect “true price” predictor. The model estimates market-based pricing using historical data.

---

## 👥 Team
- Kristal Abel
- Emily Dietz
- Thomas McClanahan
- Paulius Sateika
- Wanyao Li


---

## 🚀 Status
Project setup and experiment design in progress.


## Data
Place raw Inside Airbnb files in `data/raw/`.

Raw data files are excluded from Git tracking through `.gitignore`, so they will not appear in the GitHub repository.



## Data File Naming Convention

All team members should use the same filenames for consistency:

- listings.csv.gz
- calendar.csv.gz

These files should be placed in `data/raw/`.
