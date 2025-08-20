# 📈 E-commerce Sales Forecasting & Anomaly Detection

An end-to-end **Streamlit web application** for time-series analysis on a multi-series e-commerce dataset. This project demonstrates a full-cycle machine learning approach, from data exploration to building a scalable, validated forecasting model.

The app supports:

  * **Interactive Forecasting** → Visualize forecasts for 500 different product-store combinations.
  * **Model Validation** → View performance metrics from rigorous backtesting.
  * **Anomalies** → Identify unusual sales spikes and drops in historical data.

-----

## ⚡ My Journey: A Build Log of V1 to V2

This project is a testament to an iterative build process. Instead of a single, static solution, it evolved from a simple V1 to a robust, scalable V2. It’s a showcase of **problem-solving, persistence, and continuous improvement**.

### V1: The Foundational Build

  * **Focus:** Single-series forecasting on a small-scale dataset (`Sample - Superstore.csv`).
  * **Goal:** Prove the core concepts of forecasting (Prophet) and anomaly detection (STL + IsolationForest).
  * **Key Learnings:**
      * Debugging state-related issues in notebooks.
      * Packaging a simple analysis into a Streamlit app.
      * The importance of a clean, reproducible code block.

### V2: The Scalable Upgrade

  * **Focus:** Multi-series forecasting on a large-scale dataset (`Store Item Demand Forecasting Challenge`).
  * **Goal:** Elevate the V1 analysis to a production-ready standard.
  * **Key Challenges Overcome:**
      * **Scalability:** Built a pipeline to train and forecast **500 individual time series**.
      * **Validation:** Implemented **rolling-origin backtesting** to prove model performance across different time periods, a crucial step missing from many basic projects.
      * **Dashboarding:** Upgraded the Streamlit app to handle complex data, with interactive dropdowns and dynamic plots.

👉 This repo is not just a final version; it’s a **detailed log of the entire development process**, demonstrating how a simple idea can be engineered into a sophisticated solution.

-----

## 📂 Project Structure

```
ecom-forecast-anomaly/
├── README.md           # This project overview
├──requirements.txt    # All Python dependencies
│  
├── V1/                 # The original, foundational project
│   ├── app.py          
│   └── notebooks/
│
└── V2/                 # The advanced, scalable project
    ├── app.py          # The main Streamlit web app
    └── data/           # The V2 dataset


```

-----

## 🚀 Features

  * ✅ **Multi-Series Forecasting:** Forecasts for **500 different time series** using Prophet.
  * ✅ **Rigorous Backtesting:** Validates model accuracy with rolling-origin cross-validation.
  * ✅ **Interactive Dashboard:** Explore forecasts and performance metrics with Streamlit.
  * ✅ **End-to-End Pipeline:** All steps, from data ingestion to visualization, are in a single, runnable app.
  * ✅ **Anomalies:** Identifies unusual spikes and drops in historical sales.

-----

## 🛠️ Tech Stack

  * **Python 3.10+**
  * **Streamlit** (for the web app)
  * **Prophet** (for forecasting)
  * **Pandas, NumPy** (for data manipulation)
  * **Statsmodels, pyod** (for anomaly detection)

-----

## 📊 Demo (Example Output)

-----

## 🔧 How to Run Locally

```bash
# Clone the repository
 git clone[link:https://github.com/Dvipg/e-commerce-demand-forecasting.git]
 cd ecom-forecast-anomaly

# Install all dependencies
pip install -r requirements.txt

# Run the V2 Streamlit app
cd V2
streamlit run app.py
```

-----

## 🌍 Deployment

This project is deployable on **Streamlit Cloud** for free link: (App)[link: ]
-----

## 🏆 Key Takeaway

This project represents more than just a successful model. It demonstrates:

  * **Technical Depth:** Mastery of key data science libraries and concepts.
  * **Engineering Mindset:** The ability to build a robust, scalable, and deployable application.
  * **Problem-Solving:** The story of how an initial V1 was improved into a powerful V2 solution.

-----

## 📜 License

This project is open-source and free to use and modify.
