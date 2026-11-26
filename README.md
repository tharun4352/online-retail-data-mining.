# Online Retail Data Mining & Data Warehouse Analysis

An interactive Streamlit dashboard built around the submitted Google Colab notebook. It explores an Online Retail transaction dataset through data cleaning, customer/product analysis, a star-schema warehouse, OLAP operations and Random Forest customer churn classification.

## Features

- Upload the original Online Retail CSV or ZIP file from the sidebar.
- Sales KPIs, monthly trends and yearly revenue.
- Product and country performance.
- Customer-level purchase metrics.
- Star schema: fact sales, customer, product and date dimensions.
- OLAP roll-up, drill-down, slice, dice and pivot operations.
- Customer churn model using the notebook's 90-day target definition and Random Forest configuration.
- Download generated warehouse tables and OLAP results as CSV.
- Data quality and dimension-key validation.

## Dataset

The original notebook reads `data.csv.zip` from a Google Drive folder. The uploaded notebook does **not** embed the dataset. Download the same `data.csv` or `data.csv.zip` file from your Google Drive and upload it through the dashboard's sidebar. The app expects these columns:

`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

The public Online Retail dataset commonly used for this coursework is available from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/352/online+retail

## Run locally

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/online-retail-data-mining.git
cd online-retail-data-mining
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Select **Create app**, choose this repository, branch `main`, and main file `app.py`.
4. Deploy. Open the generated public URL and upload the dataset in the sidebar.

## Methodology notes

- Cancelled invoices (invoice numbers beginning with `C`), non-positive quantities and non-positive unit prices are excluded from valid sales, as in the notebook.
- Revenue is calculated as `Quantity * UnitPrice`.
- The warehouse fact table drops records without customer, product or date dimension keys, matching the notebook's final fact-table construction.
- Churn is labelled as `1` when a customer has no purchases in the following 90-day period and `0` otherwise.
- Random Forest settings match the notebook: 200 estimators, max depth 10, minimum split 5, minimum leaf 2, random state 42 and balanced class weights.
- Metrics are recomputed from the uploaded dataset. The notebook's saved run recorded 64.39% accuracy, 57.14% precision, 68.97% recall, 62.50% F1 and 0.7089 ROC-AUC.

## Repository structure

```text
online-retail-data-mining/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── data_mining_original.ipynb
```

## Academic integrity

This dashboard is an interface around the submitted notebook's analysis. It does not claim stock-market forecasting; the notebook's supervised learning task is customer churn classification.
