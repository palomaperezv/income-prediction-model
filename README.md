# Income Prediction Model: Census Data Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project that predicts whether an individual's annual income exceeds $50,000 based on census demographic data. Built with Random Forest classification and designed for real-world customer segmentation applications.

---

## 📊 Project Overview

**Objective:** Predict income levels (>$50K vs ≤$50K) using socioeconomic and demographic features from census data.

**Business Value:** This model enables targeted marketing campaigns, credit scoring, and customer segmentation for financial services and marketing companies.

**Key Results:**
- **Test Accuracy:** 82%
- **Minority Class Recall:** 85% (high-income earners)
- **F1-Score (>50K class):** 0.69

---

## Key Features

- **Robust preprocessing pipeline** handling missing values and high-cardinality categorical features
- **Feature engineering** with strategic grouping to reduce overfitting
- **Class imbalance handling** using balanced weights for minority class optimization
- **Production-ready inference** on unlabeled test data
- **Comprehensive EDA** with visualizations and statistical analysis

---

## 📁 Dataset

The project uses census data with the following features:

**Numerical Variables:**
- `age`: Individual's age
- `education.num`: Years of education
- `hours.per.week`: Weekly working hours
- `capital.gain`: Capital gains
- `capital.loss`: Capital losses
- `fnlwgt`: Final weight (census sampling weight)

**Categorical Variables:**
- `workclass`: Employment sector (Private, Government, Self-employed, etc.)
- `education`: Educational level achieved
- `marital.status`: Marital status
- `occupation`: Type of occupation
- `relationship`: Family relationship role
- `race`: Racial background
- `sex`: Gender
- `native.country`: Country of origin

**Target Variable:**
- `income`: Binary classification (>50K, ≤50K)

---

##  Installation & Requirements

### Prerequisites

```bash
python >= 3.8
```

### Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🚀 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/income-prediction-model.git
cd income-prediction-model
```

### 2. Run the Analysis

```python
# Execute the full pipeline
python adults_eda_basline_model.py
```

### 3. Key Workflow Steps

The script performs the following operations:

```python
# Load and preprocess data
df = pd.read_csv('train.csv')
df.replace('?', 'Unknown', inplace=True)

# Feature engineering
df = preprocess_marital_status(df)
df = preprocess_relationship(df)
df = preprocess_occupation(df)
# ... other preprocessing functions

# Train Random Forest model
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

# Make predictions on test data
predictions = rf_clf.predict(X_test)
```

### 4. Generate Predictions

The model outputs a `submission.csv` file with predicted income classes for unlabeled data:

```
ID,PRED
1,<=50K
2,>50K
3,<=50K
...
```

---

## 📈 Methodology

### 1. Exploratory Data Analysis (EDA)

- Identified hidden missing values (`?`) representing ~7% of data
- Analyzed class imbalance (75.7% ≤50K, 24.3% >50K)
- Assessed categorical cardinality and feature distributions
- Outlier analysis on continuous variables

### 2. Data Preprocessing

**Missing Value Strategy:**
- Replaced `?` with `"Unknown"` category instead of deletion
- Rationale: Missingness may carry predictive information (MNAR)

**Feature Engineering:**
- **Marital Status Grouping:** Consolidated into Married, Single, Separated
- **Relationship Grouping:** Partner vs Non-Partner hierarchy
- **Occupation & Workclass Tiering:** Professional, Services, Manual, Other
- **Country Grouping:** English-speaking, Latin America, Asia, Other, Unknown

**Encoding:**
- One-Hot Encoding for nominal categorical variables
- Binary encoding for target variable

### 3. Model Training

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- `n_estimators=100`
- `max_depth=20`
- `min_samples_split=20`
- `min_samples_leaf=10`
- `class_weight='balanced'` ← Critical for handling class imbalance
- `random_state=42`

**Rationale for Balanced Weights:**
Given the 3:1 class ratio, the model was configured to prioritize recall on the minority class (>50K), preventing it from becoming a "majority classifier."

### 4. Model Evaluation

**Performance Metrics:**

| Metric | ≤50K Class | >50K Class |
|--------|-----------|-----------|
| Precision | 0.89 | 0.58 |
| Recall | 0.80 | 0.85 |
| F1-Score | 0.84 | 0.69 |
| **Overall Accuracy** | **82%** | - |

**Confusion Matrix Insights:**
- **High Recall (85%)** for >50K class: Successfully identifies most high-income earners
- **Trade-off:** Lower precision (58%) means some false positives, but acceptable for discovery-driven marketing

---

## Feature Importance

The Random Forest model identified the following as the most influential predictors:

1. **Relationship (Partner)** - ~20% importance
2. **Marital Status (Married)** - ~20% importance  
3. **Capital Gain** - ~15% importance
4. **Education (Years)** - ~12% importance
5. **Age** - ~10% importance

**Interpretation:** Household stability and socio-demographic maturity are the strongest drivers of high income.

---

## Business Application

### Use Case: Premium Credit Card Marketing Campaign

**Model Profile:**
- **"Wide Net" Strategy:** Prioritizes recall over precision
- **Cost-Benefit:** Missing a high-value customer (FN) costs more than sending extra brochures (FP)
- **Efficiency Gain:** 2.4x better than random targeting

**Real-World Impact:**
- Marketing team receives a ranked list of prospects
- 85% of actual high-income individuals are captured
- Reduced wasted resources compared to blanket campaigns

---

## 📊 Sample Visualizations

The project includes:
- Class distribution plots
- Feature importance bar charts
- Correlation heatmaps
- Confusion matrix visualization
- Prediction distribution analysis

---

## 🗂️ Project Structure

```
income-prediction-model/
│
├── adults_eda_basline_model.py    # Main analysis script
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── submission.csv                 # Model predictions output
│
├── data/
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset (unlabeled)
│
└── notebooks/
    └── eda_exploration.ipynb     # Jupyter notebook (if applicable)
```

---

##  Future Improvements

- **Hyperparameter Tuning:** Grid search for optimal `max_features` and `min_samples_leaf`
- **Advanced Models:** Test XGBoost, LightGBM for potential performance gains
- **Feature Engineering:** Add polynomial features, interaction terms
- **Threshold Optimization:** Adjust classification threshold to fine-tune precision/recall trade-off
- **Cross-Validation:** Implement k-fold CV for more robust evaluation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Paloma Perez Valdenegro**

- GitHub: [@palomaperezv](https://github.com/palomaperezv)
- LinkedIn: [Paloma Perez V.](https://www.linkedin.com/in/paloma-perez-valdenegro-27a2621a5/)
- Email: palomprz@gmail.com

---

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Adult Census Income Dataset
- Inspired by real-world customer segmentation challenges in financial services
- Built as part of a Data Science portfolio project

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Classifier Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/)
- [Dimensionality Reduction](https://gayathri-siva.medium.com/dimensionality-reduction-feature-selection-and-feature-extraction-28c732818e96)

---

**⭐ If you found this project helpful, please consider giving it a star!**
