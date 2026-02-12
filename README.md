# Customer Transaction Prediction

A machine learning project to predict customer transaction behavior in the banking domain using supervised learning techniques.

## 📋 Project Overview

This project aims to predict whether a customer will make a transaction in the future based on anonymized numerical features. The dataset presents a binary classification problem with class imbalance, making it a challenging and realistic scenario commonly encountered in banking and financial services.

**Domain:** Banking  
**Technique:** Supervised Machine Learning (Binary Classification)  
**Project Type:** Capstone Project – PRCP-1003

## 🎯 Problem Statement

Predict whether a customer will make a transaction based on anonymized features. This involves:
- Performing comprehensive data analysis on the given dataset
- Building and evaluating predictive models to identify transaction-likely customers
- Handling class imbalance effectively
- Comparing multiple machine learning algorithms

## 📊 Dataset

- **Features:** Anonymized numerical variables (var_0 to var_199)
- **Target Variable:** Binary (0 = No Transaction, 1 = Transaction)
- **Challenge:** Highly imbalanced dataset with minority class representing transactions

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Imbalanced Learning: `imballearn` (SMOTE)

## 🔍 Methodology

### 1. Data Analysis
- Dataset overview and structure analysis
- Missing value detection
- Target variable distribution analysis
- Class imbalance identification

### 2. Data Preprocessing
- Feature and target separation (removing ID_code)
- Train-test split (80-20) with stratification
- Feature scaling using StandardScaler
- SMOTE application for handling class imbalance

### 3. Model Development
Three classification algorithms were implemented and compared:

#### Logistic Regression
- Max iterations: 1000
- Best overall performance for imbalanced data

#### Random Forest Classifier
- n_estimators: 120
- max_depth: 14
- min_samples_split: 15
- min_samples_leaf: 10
- Optimized for parallel processing (n_jobs=-1)

#### Decision Tree Classifier
- max_depth: 10
- min_samples_split: 2
- min_samples_leaf: 1

### 4. Model Evaluation
Models were evaluated using metrics appropriate for imbalanced datasets:
- **Accuracy**
- **Precision**
- **Recall** (Critical for identifying transaction customers)
- **F1 Score**
- **ROC-AUC** (Primary metric for model comparison)
- **Confusion Matrix**

## 📈 Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | ~0.81 | ~0.29 | **0.76** | ~0.42 | **0.86** |
| Random Forest | ~0.83 | ~0.44 | ~0.26 | ~0.33 | ~0.76 |
| Decision Tree | Lower | Lower | Lower | Lower | Lower |

### Key Findings

- **Logistic Regression** performed best with the highest **Recall (0.76)** and **ROC-AUC (0.86)**, making it the most suitable model for identifying transaction customers in an imbalanced dataset
- **Random Forest** achieved higher accuracy but showed low recall, indicating poor detection of minority class customers
- **Decision Tree** showed the weakest overall performance

### Confusion Matrix Insights (Logistic Regression)
- **True Negatives:** 28,519 (Correctly identified non-transaction customers)
- **True Positives:** 3,059 (Correctly identified transaction customers)
- **False Positives:** 7,461 (Non-transaction customers misclassified as transaction)
- **False Negatives:** ~961 (Transaction customers missed)

The model successfully learns patterns from both classes, with high recall ensuring most transaction customers are identified.

## 💡 Key Insights

1. **Class Imbalance Challenge:** The dataset is highly imbalanced, requiring SMOTE to balance the training data
2. **Metric Selection:** Traditional accuracy is misleading for imbalanced datasets; Recall and ROC-AUC are more appropriate
3. **Model Selection:** Logistic Regression outperformed ensemble methods for this specific imbalanced classification task
4. **Business Impact:** High recall (76%) ensures the bank identifies most customers likely to make transactions, minimizing missed opportunities

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/vmurugavel001-oss/customer-transaction-prediction.git
cd customer-transaction-prediction
```

2. Ensure your dataset (`data.csv`) is in the project directory

3. Open and run the Jupyter notebook:
```bash
jupyter notebook Customer_transaction.ipynb
```

## 📁 Project Structure

```
customer-transaction-prediction/
│
├── Customer_transaction.ipynb    # Main analysis notebook
├── data.csv                       # Dataset (not included)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies (optional)
```

## 🔧 Challenges Faced

- **Anonymized Features:** Limited domain-based interpretation and feature understanding
- **Computational Complexity:** Training on large dataset increased computation time, especially after SMOTE
- **Metric Selection:** Choosing appropriate evaluation metrics for class imbalance

### Solutions Applied
- SMOTE for balancing training data
- Ensemble and linear models comparison
- Focus on Recall and ROC-AUC over accuracy

## 📝 Conclusion

This project successfully demonstrates the application of supervised machine learning to predict customer transaction behavior in banking. Through comprehensive analysis and model comparison, **Logistic Regression** emerged as the optimal solution, achieving 76% recall and 0.86 ROC-AUC score. The model effectively balances precision and recall, making it suitable for real-world deployment where identifying potential transaction customers is crucial for business success.

## 🔮 Future Enhancements

- Feature engineering using domain knowledge (if features are de-anonymized)
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Ensemble methods combining multiple models
- Cost-sensitive learning approaches
- Deep learning models for comparison

## 👤 Author

murugavel   
- GitHub: [@vmurugavel001-oss](https://github.com/vmurugavel001-oss)
- LinkedIn: [murugavel-v](https://www.linkedin.com/in/murugavel-v)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

