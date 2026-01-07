import pandas as pd
from prophet import Prophet
import joblib
import os

def prepare_time_series(df_clean):
    """Transforms raw retail data into Prophet-ready format."""
    # [cite_start]Resample to Monthly Total Sales [cite: 128, 129]
    sales_ts = (df_clean.set_index('InvoiceDate')
                .resample('ME')['TotalPrice']
                .sum())
    
    # [cite_start]Prophet requires columns named 'ds' and 'y' [cite: 546, 547]
    prophet_df = sales_ts.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # [cite_start]Ensure datetime is naive for Prophet [cite: 105, 546]
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    return prophet_df

def train_and_save_forecast(prophet_df, model_path='models/prophet_model.pkl'):
    """Trains Prophet model and saves for future inference."""
    # [cite_start]Initialize and fit the model [cite: 548, 549]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    # [cite_start]Generate future dates for 6 months [cite: 550]
    future = model.make_future_dataframe(periods=6, freq='ME')
    
    # [cite_start]Predict [cite: 551]
    forecast = model.predict(future)
    
    # [cite_start]Save results and model [cite: 595, 597]
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('data/sales_forecast.csv', index=False)
    joblib.dump(model, model_path)
    
    print("Prophet model and forecast saved successfully.")
    return forecast