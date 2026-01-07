import os
import sys
import argparse
from src.train import load_and_clean_data, perform_rfm_segmentation, train_pipeline
from src.forecast import prepare_time_series, train_and_save_forecast

def main(data_path):
    # Create directory for artifacts if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    print("--- Stage 1: Data Ingestion & Cleaning ---")
    df_clean = load_and_clean_data(data_path)
    print(f"Cleaned data shape: {df_clean.shape}") # [cite: 125]

    print("\n--- Stage 2: Feature Engineering (RFM) ---")
    rfm = perform_rfm_segmentation(df_clean) # [cite: 319]
    rfm.to_csv('data/processed/rfm_data.csv') # [cite: 598]

    print("\n--- Stage 3: Training Churn Model ---")
    # This function saves 'rf_churn_model.pkl' and 'scaler.pkl'
    train_pipeline(data_path) # 

    print("\n--- Stage 4: Time Series Forecasting ---")
    prophet_df = prepare_time_series(df_clean) # [cite: 546]
    train_and_save_forecast(prophet_df) # [cite: 591, 597]

    print("\nPipeline Complete. All artifacts saved to /models and /data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Intelligence Pipeline")
    parser.add_argument("--data", type=str, default="data/online_retail_II.csv", help="Path to raw CSV")
    args = parser.parse_args()
    
    main(args.data)