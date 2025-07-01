# Customer Churn Prediction & Driver Analysis

## Project Overview

This project addresses the critical business challenge of **customer churn** within a telecommunications context. For any subscription-based company, understanding why customers leave is paramount to sustaining revenue and fostering growth. My objective was to develop a robust predictive model that not only accurately forecasts churn but also identifies the **key factors driving both customer retention and attrition**, thereby enabling the development of highly targeted intervention strategies.

## Data Source & Technologies

* **Data Source:** This analysis utilizes the "Customer Churn Prediction: Analysis," a synthetic dataset publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
* **Technologies Used:**
    * **Python:** The primary programming language for all analytical tasks.
    * **Pandas:** Essential for efficient data manipulation, cleaning, and analysis.
    * **NumPy:** Utilized for numerical operations and array computing.
    * **Scikit-learn:** The core library for machine learning model development, including Logistic Regression and various preprocessing techniques.
    * **Matplotlib:** Employed for fundamental data visualization.
    * **Seaborn:** Used for creating enhanced, aesthetically pleasing statistical graphics.

## Methodology

The project workflow was structured through several key analytical stages to ensure a comprehensive and robust solution:

1.  **Data Preparation:**
    Initial steps focused on ensuring data quality. This involved addressing missing values, converting data types to appropriate formats, and performing general data cleaning to prepare the dataset for analysis.

2.  **Feature Engineering:**
    This crucial phase transformed raw data into more insightful features, enhancing the model's predictive power and interpretability. Key techniques included:
    * **Grouping Continuous Variables:** Categorizing numerical features like customer age (e.g., into '25-39', '40-54' ranges) and tenure (e.g., 'under 1 year', '1-3 years') into distinct, more interpretable groups.
    * **Creating Composite Features:** Developing new variables such as `Services_Count` (representing the total number of core services subscribed to by a customer).
    * **Developing Interaction Terms:** Constructing specific features (e.g., an interaction term between contract type and tenure) to understand how certain factors, like customer tenure, specifically impact those on a month-to-month contract.

3.  **Model Development:**
    A **Logistic Regression** model was chosen and built for the churn prediction task. This model was selected primarily for its strong interpretability, allowing for clear understanding of the influence of each feature on churn probability.

4.  **Feature Importance Analysis:**
    Following model training, a thorough analysis of the model's coefficients was conducted. This quantitative assessment identified the most influential drivers of both customer churn and retention, providing actionable insights.

## Key Findings & Outcome

The developed Logistic Regression model demonstrated exceptional performance in predicting customer churn:

### Model Performance
* **Overall Accuracy:** 98.5%
* **Precision (Churners):** 100% (indicating nearly all predicted churners actually churned, minimizing false positives)
* **Recall (Churners):** 98% (meaning the model correctly identified almost all actual churners, minimizing false negatives)

### Key Drivers of Customer Churn
Analysis of feature importance revealed the following dominant factors pushing customers towards attrition:

* **Month-to-Month Contracts:** Customers on these flexible contracts are highly prone to churn. This risk significantly intensifies the longer they remain on this plan, indicating a compounding negative effect of contract type and tenure.
* **High Monthly Charges:** A strong indicator of churn, suggesting that customers may perceive a lack of value for the services received relative to their cost, or are sensitive to pricing.
* **Lack of Tech Support:** Customers who do not subscribe to or utilize technical support services are significantly more likely to churn, underscoring the critical importance of robust customer assistance and problem resolution.
* **New Customers (Under 1 Year Tenure):** This segment exhibits a higher inherent churn risk, highlighting the critical need for proactive engagement and onboarding strategies during the initial customer lifecycle phase.

### Key Drivers of Customer Retention
Conversely, the analysis identified factors strongly associated with customer loyalty and retention:

* **Tech Support Subscriptions:** Customers actively subscribing to technical support demonstrate significantly higher retention rates, emphasizing the value of reliable support.
* **Longer-Term Contracts (One-Year/Two-Year):** These contract types provide substantial stability and effectively reduce churn compared to flexible month-to-month options.
* **Mid-to-Long Term Tenure:** Customers who have stayed for 1-3 years or over 3 years exhibit strong loyalty, becoming less likely to churn over time, indicating a period where retention efforts yield high returns.

### Deeper Dive: The Nuance of Internet Service

Beyond the primary churn drivers, an intriguing and somewhat counter-intuitive pattern emerged concerning **Internet Service Type**:

While customers who *only* subscribe to phone service (i.e., no internet service through our company) show a higher propensity to stay, customers *with* any form of our internet service (including both Fiber Optic and DSL) exhibit a minor but noticeable increased churn risk. This finding is significant as it suggests customers might be finding more reliable, faster, or cheaper internet options elsewhere, making our internet service a potential point of vulnerability in a competitive market. This warrants a critical re-evaluation of our internet service's value proposition.

## Recommendations

Based on these comprehensive insights, the following strategic recommendations are advised for the telecom company to mitigate churn and enhance customer retention:

1.  **Target Month-to-Month Customers Proactively:**
    Implement proactive retention campaigns specifically for month-to-month customers, especially as they approach the 6-12 month mark. These campaigns should highlight the long-term benefits of commitment, offer exclusive incentives for contract upgrades, and emphasize the value of bundled services.

2.  **Enhance Tech Support & Contract Promotion:**
    More aggressively promote the value and benefits of Tech Support subscriptions and longer-term contract options (One-Year, Two-Year). This could involve showcasing positive customer testimonials, offering introductory discounts on support services, or integrating support as a perceived value-add in higher-tier plans.

3.  **Investigate Internet Service Competitiveness:**
    Conduct a deeper, urgent analysis into customer satisfaction with current internet services and a thorough competitive landscape analysis of competitor pricing, reliability, and features. If internet service is indeed a churn driver, it likely points to external market pressures or perceived service gaps that need immediate attention.

## Repository Structure

This repository is organized to provide clear access to all project components:

Customer_Churn_Prediction_Analysis/
├── notebooks/                 (Jupyter notebooks detailing the analysis workflow)
│   └── churn_prediction_analysis.ipynb
├── src/                       (Optional: Python scripts for reusable functions or model code)
│   └── data_preprocessing.py
│   └── model_training.py
├── data/                      (Contains the raw and/or processed dataset)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                    (Optional: Saved trained models, e.g., .pkl files)
│   └── logistic_regression_model.pkl
├── README.md                  (This file: project overview, methodology, findings, and recommendations)
├── requirements.txt           (Lists all Python dependencies)


## How to Run the Project

To replicate and explore this project locally:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/Customer_Churn_Prediction_Analysis.git](https://github.com/YOUR_GITHUB_USERNAME/Customer_Churn_Prediction_Analysis.git)
    cd Customer_Churn_Prediction_Analysis
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file in your project's root directory that lists all the Python libraries used, e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)*
4.  **Run the Analysis:**
    * If using Jupyter Notebooks:
        ```bash
        jupyter notebook
        ```
        Then open `notebooks/churn_prediction_analysis.ipynb` in your browser.
    * If using Python scripts:
        ```bash
        python src/model_training.py
        ```

---
**Thank you for exploring this project!**
