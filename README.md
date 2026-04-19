# Airbnb Pricing Recommendation Project (Columbus, Ohio)

## 📌 Project Overview
This project develops an end-to-end machine learning system to predict and recommend optimal nightly prices for Airbnb listings in Columbus, Ohio.
The system leverages listing attributes and market data to generate data-driven pricing recommendations for hosts and property managers. 
In addition to model development, the project emphasizes reproducibility, scalability, and deployment by integrating version control, structured workflows, and a user-facing application.

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

- ## How to Run the Project

1. Open the notebooks in Google Colab
2. Upload the dataset (listings.csv.gz)
3. Run all cells in order
4. View model outputs and predictions

## Models Implemented
- Linear Regression
- Random Forest (best performing)

These files should be placed in `data/raw/`.

Added prediction function for model inference
