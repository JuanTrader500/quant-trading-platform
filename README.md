# SP500 MLOps Project

## 📌 Project Overview
This project aims to develop a professional and scalable MLOps pipeline for financial asset analysis and prediction (specifically for SP500 and related indices), following the **CRISP-DM** methodology.

## 🛠 Methodology: CRISP-DM
1. **Business Understanding**: Define trading goals and success metrics.
2. **Data Understanding**: Exploratory Data Analysis (EDA) of financial assets.
3. **Data Preparation**: Professional ETL pipeline for data cleaning and feature engineering.
4. **Modeling**: Implement and compare ML models (e.g., Ridge Regression).
5. **Evaluation**: Financial and statistical validation of model performance.
6. **Deployment**: Production-ready pipeline for inference.

## 📂 Project Structure
- `data/`: Raw and processed financial data.
- `src/`: Core source code.
    - `data_pipeline/`: Data extraction and ingestion logic.
    - `features/`: Data cleaning and feature engineering.
    - `models/`: Model training, evaluation, and registration.
    - `utils/`: Helper functions and shared utilities.
- `tests/`: Unit and integration tests for CI/CD.
- `config/`: Configuration files for tickers and hyperparameters.
- `notebooks/`: Prototyping and EDA.

## 🚀 CI/CD Pipeline
- Integration tests run automatically to ensure data pipeline stability.
- Code quality checks (Linting).

## 📈 Current Status
- [x] Project structure initialization.
- [x] Data migration.
- [ ] Pipeline refactoring.
- [ ] Feature engineering modularization.
- [ ] Model training scripts implementation.
- [ ] CI/CD setup.
