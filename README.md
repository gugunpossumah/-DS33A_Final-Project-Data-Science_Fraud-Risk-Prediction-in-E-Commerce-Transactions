# Fraud Risk Prediction in E-Commerce Transactions

## 📋 Project Overview

This project focuses on building a machine learning system to detect fraudulent e-commerce transactions. As digital transactions continue to grow, the risk of fraudulent activities also increases, causing significant financial losses and damaging customer trust. This solution uses advanced data science techniques to identify suspicious patterns and help e-commerce platforms prevent fraud effectively.

## 🎯 Business Problem

E-commerce companies face challenges in:
- Identifying potentially fraudulent transactions among millions of daily transactions
- Reducing financial losses without disrupting legitimate customer experiences
- Balancing between false positives (blocking legitimate transactions) and false negatives (missing actual fraud)

## 📊 Dataset

The project uses the **Fraudulent E-Commerce Transaction Data** from Kaggle, containing:
- **1.47 million+ transactions** with 21 features
- **5% fraud rate** (highly imbalanced data)
- Features include transaction amount, payment method, device used, customer age, account age, and more

**Dataset Source**: [Fraudulent E-Commerce Transaction Data on Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data)

## 🛠️ Technical Approach

### Methodology: CRISP-DM Framework
1. **Business Understanding** - Identifying fraud detection needs and business impact
2. **Data Understanding** - Exploratory analysis of transaction patterns
3. **Data Preparation** - Cleaning, feature engineering, and handling class imbalance
4. **Modeling** - Training multiple machine learning algorithms
5. **Evaluation** - Comparing model performance metrics
6. **Deployment** - Streamlit app for real-time predictions

### Key Features Engineered
- **Time-based features**: Transaction hour, day of week, weekend indicator, night transactions
- **Behavioral features**: New customer flag, large transaction indicator
- **Security features**: Address mismatch, IP address analysis
- **Financial features**: Amount per item, transaction amount analysis

### Models Implemented
- **Logistic Regression** - Baseline model for interpretability
- **Decision Tree** - For capturing non-linear relationships
- **Random Forest** - Ensemble method for improved accuracy
- **XGBoost** - Advanced gradient boosting for optimal performance

## 📈 Model Performance Comparison

| Metric | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|--------|---------------------|---------------|---------------|---------|
| **Accuracy (Test)** | 85% | 88% | 84% | 87% |
| **Precision (Class 1 - Fraud)** | 20% | 23% | 19% | 22% |
| **Recall (Class 1 - Fraud)** | 61% | 58% | 61% | 59% |
| **F1-Score (Class 1)** | 0.3506 | 0.3536 | 0.3488 | 0.3604 |
| **AUC Score** | 0.7669 | 0.7726 | 0.7670 | 0.7735 |
| **Best Threshold** | 0.7902 | 0.8205 | 0.6966 | 0.7726 |
| **Precision (Optimal)** | 27.34% | 32.38% | 28.50% | 30.42% |
| **Recall (Optimal)** | 48.85% | 38.95% | 44.93% | 44.19% |
| **Overfitting Level** | Medium | Low | High | Medium |
| **Recommendation** | For high recall | For high precision | - | **Best Model** |

## 🏆 Model Selection

**XGBoost emerged as the best-performing model** with the highest F1-score (0.3604) and AUC score (0.7735), making it the most balanced model for fraud detection. The model shows good generalization with medium overfitting and provides the best trade-off between precision and recall.

### Model Recommendations:
- **Logistic Regression**: Choose when high recall is prioritized (catching most fraud cases)
- **Decision Tree**: Optimal when high precision is needed (minimizing false positives)
- **XGBoost**: Best overall performance for balanced fraud detection

## 🚀 How to Use the Streamlit App

### Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, scikit-learn, xgboost, streamlit

### Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/fraud-detection-ecommerce.git
cd fraud-detection-ecommerce
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

### Using the Prediction Interface
1. **Input Transaction Details**: Fill in the form with transaction information:
   - Transaction amount and quantity
   - Payment method and product category
   - Customer age and account age
   - Device used and transaction time
   - Shipping and billing details

2. **Get Prediction**: Click the "Predict Fraud Risk" button to analyze the transaction

3. **Interpret Results**: 
   - **Green**: Low risk - Legitimate transaction
   - **Red**: High risk - Potential fraud
   - The app provides confidence scores and key factors contributing to the prediction

## 💡 Key Insights

### Fraud Patterns Identified
- **Time patterns**: Higher fraud rates during night hours (12 AM - 6 AM)
- **Account behavior**: New accounts (<30 days) show higher fraud probability
- **Transaction patterns**: Large transactions and address mismatches are strong indicators
- **Payment methods**: Certain payment methods show higher fraud rates

### Business Impact
- **Cost savings**: Potential to prevent millions in fraudulent transactions
- **Customer trust**: Reduced false positives maintain customer satisfaction
- **Operational efficiency**: Automated screening reduces manual review workload

## 🔮 Future Enhancements

- Real-time API integration with payment gateways
- Adaptive learning to handle evolving fraud patterns
- Enhanced feature engineering with additional data sources
- Dashboard for monitoring fraud trends and model performance

## 📚 References

- CRISP-DM methodology framework
- IEEE research papers on fraud detection systems
- Kaggle data science competitions on fraud prediction

## 👥 Author

**Gulbuddin Ikhmatiar Possumah**  
Data Science Enthusiast | DS33A Cohort

---

*This project demonstrates the power of machine learning in solving real-world business problems in the e-commerce industry. The solution balances detection accuracy with customer experience, providing a practical approach to fraud prevention.*
