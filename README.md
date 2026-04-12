# London Property Price Prediction System
An AI-powered property price predictor that accurately predicts house prices in London, accounting for many factors including but not limited to school quality, transport accessibility and local crime rates.

## Overview
This project proposes the development of a machine learning-based system designed to predict property prices within a local area. The system utilises supervised learning techniques to analyse historical housing data and identify relationships between property prices and key influencing factors.

The model takes multiple input variables, including the number of bedrooms and bathrooms, distance to the city centre, school ratings, crime rates, and public transport accessibility. These variables were selected as they represent both structural and environmental factors that significantly impact property values.

The output of the system is a predicted property price, allowing users to obtain more accurate and data-driven estimates compared to traditional methods. By leveraging machine learning, the system is able to capture complex patterns in the data and improve prediction accuracy.

## Problem Statement
One of the key challenges in the housing market is the difficulty in determining accurate property prices at a local level. Prices are influenced by a combination of factors such as neighbourhood quality, access to public transport, school performance, and crime rates.

Existing estimation methods are often slow, generic, and lack the ability to incorporate multiple influencing variables simultaneously. This creates a gap for a more efficient and accurate system that can provide realistic property price predictions based on real-world data.

Therefore, there is a need for a data-driven solution that can analyse multiple factors and generate reliable property price estimates tailored to specific locations.

## Data Sources
- London Property Dataset (Property Price, Room Counts, Floor Area)
- ONS Postcode Directory (Postcodes and Latitude/Longitude)
- Ofsted School Inspection Dataset (School Inspection Ratings)
- UK Crime Dataset (Local Crime Data)
- National Public Transport Access Nodes (Public Transportation Stops)

## Methodology
### Data Cleaning
- a

### Feature Engineering
- a

### Model Development
- **Models Considered:** Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Model Used:** CatBoost
- **Target Variable:** Log-Transformed Property Price
- **Train/Test Split:** 80/20
- **Parameters**
  - **Iterations:** 1000
  - **Learning Rate:** 0.1
  - **Depth:** 8
  - **L2 Leaf Regularisation:** 5
  - **Subsample:** 0.6
  - **Random State:** 42
  - **Verbose:** 0

## Results
- Base Model with Default Parameters
  - MAE: ~0.175
  - RMSE: ~0.247
  - R²: ~0.840
- Tuned Model
  - MAE: ~0.172
  - RMSE: ~0.244
  - R²: ~0.844

### Key Insights
- a

## Demo
A user-friendly demo notebook ("app/property_price_prediction.ipynb") is provided to simulate the real-world implementation of this system.

To obtain predicted property price, the user can input:
- Postcode
- Bedroom/Bathroom/Living Room Count
- Floor Area (in sqm)
- Property Type
- Tenure (Leasehold/Freehold)
- Energy Rating (A-G)

After filling all input fields and clicking the "Predict" button, the system will generate and display the predicted property price based on its details provided by the user.

## Project Structure
a

## Installation
To install everything required to run the notebooks and scripts in this project locally, users can run the following command beforehand.
```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook "app/property_price_prediction.ipynb"
2. Run the cell
3. Enter details of the property and click the "Predict" button

## Limitations
- Due to certain technical setbacks, some files from the initial data cleaning were lost and unable to be recovered, but this does not effect the final product/implementation.
- Due to GitHub repository size limits, the original ONS Postcode Directory dataset is omitted from this repository due to its file size being too large.
- a

## Recommendations for Future Work
- a

## Contributors
- **Jonathan:** Contributed to both data preparation and modelling. Performed additional data cleaning on top of the initial preprocessing to ensure consistency across datasets. Led feature engineering by transforming and combining datasets into a unified structure. Prepared training and testing datasets and worked closely with Myo during the modelling phase. Implemented the tuned model into an interactive interface where users can easily input their house’s postcode and structural characteristics and obtain predicted prices. Also contributed to model comparison and was vital to the final model choice decision-making process.
- **Anuf:** Coordinated the project along with Jonathan and also led evaluation and report writing, including performance analysis, critical assessment, and all the documentation. Additionally, cross team collaboration supports by assisting in verifying data consistency and contributing feedback to processing, modelling and evaluation stages, and was a part of the early data gathering phase. Compared different models and contributed to model tuning.
- **Cliff:** Supported the report development and evaluation phase alongside Anuf by assisting in structuring the report, reviewing analysis, and validating the interpretation of results. Also contributed feedback across other stages to ensure consistency and data gathering. Played a key role in model comparison.
- **Myo:** Was mainly focused on initial model development and training using the prepared datasets. Collaborated with Jonathan to understand feature design and ensure appropriate model selection. Also contributed to refining output based on feedback from the evaluation stage.
- **Kai:** Led the initial data collection and preprocessing using Python and Pandas, including handling missing values, removing duplicates, standardizing column formats, and selecting relevant features. Combined multiple datasets into a single structured dataset. Also collaborated with the modelling team to interpret results and ensure the evaluation aligned with the implemented models.
- **Su:** Worked alongside Kai on data collection and preprocessing, contributing to data cleaning, formatting, and consolidation of datasets. Additionally, supported cross team collaboration by assisting in verifying data consistency and contributing feedback to both modelling and evaluation stages. Also supported later stages by ensuring data quality aligned with modelling requirements.
