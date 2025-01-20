# Fraud Detection Using Machine Learning

## Overview
This project aims to develop a robust fraud detection model using various machine learning techniques, with an emphasis on Exploratory Data Analysis (EDA) and preprocessing. The dataset contains transaction records, and the goal is to identify fraudulent transactions effectively.

## Key Features
- Comprehensive **Exploratory Data Analysis** to understand the dataset.
- Detection and handling of outliers using **Boxplots** and the **Interquartile Range (IQR)** method.
- Feature engineering for new insights, including balance difference calculations.
- Addressed class imbalance using **Synthetic Minority Over-sampling Technique (SMOTE)**.
- Multiple machine learning models:
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **Decision Tree Classifier**
- Evaluation using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.

## Dataset
The dataset contains transactional data with features such as:
- `amount`: Transaction amount.
- `oldbalanceOrg` and `newbalanceOrig`: Original and new balances for the origin account.
- `oldbalanceDest` and `newbalanceDest`: Original and new balances for the destination account.
- `type`: Transaction type (e.g., transfer, cash-out).
- `isFraud`: Target variable indicating if the transaction was fraudulent.

You can download the dataset using the following link:
[Download Dataset](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)

## Steps Performed
### 1. **Data Cleaning**
- Verified the absence of missing values.
- Identified and handled outliers in numerical columns (`amount`, `oldbalanceOrg`, etc.) using IQR.

### 2. **EDA**
- Visualized the correlation matrix to understand relationships between variables.
- Analyzed the distribution of the target variable (`isFraud`) to observe class imbalance.

### 3. **Feature Engineering**
- Created new features like `balanceOrg_diff` and `balanceDest_diff` to capture balance differences.
- Encoded categorical columns (`type`, `nameOrig`, `nameDest`) using **Label Encoding**.

### 4. **Model Training**
- Split the dataset into training and testing sets with stratification.
- Applied **SMOTE** to balance the classes in the training set.
- Trained three machine learning models:
  - **Random Forest Classifier**: Highlighted for its high accuracy and feature importance.
  - **Logistic Regression**: Tested for baseline comparison.
  - **Decision Tree Classifier**: For interpretability.

### 5. **Model Evaluation**
- Generated confusion matrices and classification reports.
- Plotted **ROC-AUC curves** for performance visualization.
- Identified key features contributing to fraud detection using feature importance.

## Results
### Model Performance
| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Random Forest          | ~90%    | High      | High   | High     | High    |
| Logistic Regression    | Moderate | Moderate  | Moderate| Moderate | Moderate|
| Decision Tree Classifier | High   | High      | High   | High     | High    |

### Key Features
The following features were most impactful in identifying fraudulent transactions:
- `balanceOrg_diff`
- `amount`
- `newbalanceDest`
- `type_encoded`

## Visualizations
- **Boxplots** of numerical features before and after outlier handling.
- **Correlation heatmap** to explore relationships between features.
- **ROC curves** for each model to visualize classification performance.

## Future Work
- Implement real-time fraud detection systems.
- Explore deep learning techniques like **LSTMs** for sequential data analysis.
- Integrate additional features such as user behavior patterns for improved detection.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/raviraj-441/fraud_detection_assignment.git
   ```

2. Navigate to the project directory:
   ```bash
   cd fraud-detection
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook:
   Execute the notebook cells step-by-step to replicate the analysis.

## Dependencies
- Python 3.11.5
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

## Acknowledgments
Special thanks to the creators of the dataset and the open-source libraries used in this project.
