import pandas as pd
import numpy as np
import joblib
import os 
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def load_and_clean_data(file_path):
    """Downloads, loads, and cleans the retail dataset."""
    path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
    files = os.listdir(path)
    csv_path = os.path.join(path, files[0]) 
    
    df = pd.read_csv(csv_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    df_clean = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    return df_clean

def perform_rfm_segmentation(df_clean):
    """Aggregates data into Recency, Frequency, and Monetary metrics."""
    reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_clean.dropna(subset=['Customer ID']).groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def train_pipeline(data_path):
    """Executes the full training workflow and saves artifacts."""
    data = load_and_clean_data(data_path)
    rfm = perform_rfm_segmentation(data)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    
    X = rfm[['Recency', 'Frequency', 'Monetary', 'Cluster']]
    y = rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_scaled = scaler.fit_transform(X_train)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Training complete. Artifacts saved to models/")

if __name__ == "__main__":
    train_pipeline('data/online_retail_II.csv')