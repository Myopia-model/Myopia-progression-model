# MyoTrack: Myopia Progression Predictor

MyoTrack is a machine learning web application that estimates how a user's spherical equivalent refractive error may change over time based on age, family history, screen time, near-work habits, and outdoor activity. The project uses a regression model to generate a projected myopia progression curve from the user's current age up to age 25.

The goal of this project is to explore how lifestyle and genetic risk factors can be used in a predictive model and presented through a simple interactive dashboard.

## Overview

Myopia, commonly known as nearsightedness, can progress during childhood and adolescence. Several factors may influence this progression, including genetics, time spent on near-work activities, screen exposure, and outdoor activity.

This application allows users to enter basic profile information and daily habit estimates. The model then predicts spherical equivalent refractive error values over future ages and visualizes the trend using an interactive chart.

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a replacement for professional eye care.

## Features

- Interactive Streamlit web interface
- User input for age, gender, parental myopia, screen time, reading/studying time, and outdoor activity
- Feature engineering for screen time, outdoor time, genetic risk, and age-based interaction terms
- Linear regression model trained using scikit-learn
- Prediction of SPHEQ values from the user's current age to age 25
- Visualization of predicted SPHEQ progression using Plotly
- Simple risk-level classification based on lifestyle and family history factors
- Clinical-style notes explaining the user's risk profile

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly

## Project Structure

```text
Myopia-progression-model/
│
├── backend/
│   └── backend.py
│
├── frontend/
│   └── frontend.py
│
├── database/
│   └── myopia.csv
│
├── src/
│   └── model1.py
│
├── .gitignore
└── README.md
```

## Dataset

The project uses a CSV dataset containing myopia-related variables such as age, gender, spherical equivalent refractive error, parental myopia history, screen-related habits, reading/studying time, and sports or outdoor activity.

The target variable used by the model is:

```text
SPHEQ
```

SPHEQ represents spherical equivalent refractive error, measured in diopters.

## Model Approach

The backend prepares the raw dataset by selecting relevant columns, converting values into numeric format, handling missing values, and creating additional engineered features.

The main features used for prediction include:

```text
AGE
SCREEN_TIME
OUTDOOR_TIME
AGE_x_SCREEN
AGE_x_OUTDOOR
GENETIC_RISK
GENDER
```

### Engineered Features

`SCREEN_TIME` combines TV time, computer/phone time, and weighted reading/studying time.

`OUTDOOR_TIME` is based on sports or outdoor activity hours.

`GENETIC_RISK` is calculated from whether the mother and/or father are myopic.

`AGE_x_SCREEN` and `AGE_x_OUTDOOR` are interaction features used to capture how screen exposure and outdoor activity may affect progression differently as age changes.

The model uses Linear Regression from scikit-learn to predict SPHEQ.

## How It Works

1. The application loads the myopia dataset.
2. The backend cleans and prepares the data.
3. A linear regression model is trained on the processed features.
4. The user enters their profile and lifestyle information in the Streamlit interface.
5. The model predicts SPHEQ values from the current age to age 25.
6. The frontend displays the projected values, progression change, risk level, and chart.

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/Myopia-model/Myopia-progression-model.git
cd Myopia-progression-model
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

Run the Streamlit application:

```bash
streamlit run frontend/frontend.py
```

## Example Use Case

A user enters their current age, gender, parental myopia history, daily screen time, reading/studying time, and outdoor activity. The application then generates a projected SPHEQ curve showing how their refractive error may change by age 25 under the current lifestyle assumptions.

The result includes:

- baseline predicted SPHEQ
- projected change by age 25
- risk level
- progression graph
- short explanatory notes

## Limitations

This project is not a clinical tool. The predictions are based on a simple machine learning model and the available dataset, so results should be interpreted carefully.

Current limitations include:

- The model uses linear regression, which may not fully capture complex medical or biological relationships.
- The dataset size and quality limit the reliability of predictions.
- The app does not account for clinical treatments such as atropine, orthokeratology, contact lenses, or other myopia-control interventions.
- Lifestyle inputs are self-reported estimates.
- The model should not be used for diagnosis or treatment decisions.

## Future Improvements

Possible improvements include:

- Add model comparison with Random Forest, Gradient Boosting, or other regression models
- Save and load the trained model instead of retraining during prediction
- Add more detailed model evaluation metrics
- Improve the frontend design and user experience
- Add data visualizations for dataset exploration
- Add unit tests for backend functions
- Deploy the application online
- Add clearer documentation for the dataset and feature meanings
- Add confidence intervals or uncertainty estimates for predictions

## Contributors

- Karanveer Singh
- Pravit
- Ajaypal

## Disclaimer

MyoTrack is for educational and research purposes only. It is not intended to diagnose, treat, or prevent any medical condition. Users should consult a licensed optometrist or ophthalmologist for professional eye care advice.
