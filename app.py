import sqlite3
import pandas as pd
import numpy as np
import os
import io
import joblib
import warnings
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Transformer Digital Twin System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main { 
        background-color: #F0F9FF; 
    }
    
    .stButton>button {
        background-color: #2563EB;
        color: #FFFFFF;
        border-radius: 6px;
        border: 1px solid #1D4ED8;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
        color: #FFFFFF;
        border: 1px solid #1E3A8A;
    }
    
    .sap-header {
        background-color: #0F172A;
        padding: 18px 24px;
        color: #F0F9FF;
        font-size: 22px;
        font-weight: 700;
        border-bottom: 4px solid #3B82F6;
        margin-bottom: 24px;
        border-radius: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .sap-title {
        background-color: #1E293B;
        padding: 12px 18px;
        color: #E0F2FE;
        font-size: 16px;
        font-weight: 600;
        border-left: 4px solid #60A5FA;
        margin-bottom: 18px;
        border-radius: 4px;
    }
    
    .insight-card {
        background-color: #FFFFFF;
        border-left: 5px solid #3B82F6;
        padding: 18px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
        border-top: 1px solid #E0F2FE;
        border-right: 1px solid #E0F2FE;
        border-bottom: 1px solid #E0F2FE;
    }
    
    .insight-label {
        font-size: 12px;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .insight-value {
        font-size: 16px;
        color: #0F172A;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "dga_database.db"
CSV_SEED_FILE = "Data_Uji_Trafo.csv"
JALUR_MODEL_LOKAL = 'model_dga_7classes_v2.pkl'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Trafo TEXT NOT NULL,
            Tanggal_Uji TEXT NOT NULL,
            H2 REAL, CH4 REAL, C2H6 REAL, C2H4 REAL, C2H2 REAL,
            CO REAL, CO2 REAL, O2 REAL, N2 REAL,
            BDV REAL, Acid REAL, Water REAL, IFT REAL,
            Status_Pemurnian TEXT,
            Is_Anomali TEXT DEFAULT 'No'
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM tabel_master")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(CSV_SEED_FILE):
        df_csv = pd.read_csv(CSV_SEED_FILE)
        df_csv['Tanggal_Uji'] = pd.to_datetime(df_csv['Tanggal_Uji']).dt.strftime('%Y-%m-%d')
        df_csv['Status_Pemurnian'] = df_csv['Status_Pemurnian'].fillna('Normal')
        df_csv['Is_Anomali'] = 'No'
        df_csv = df_csv.where(pd.notnull(df_csv), None)

        records = []
        for _, row in df_csv.iterrows():
            records.append((
                str(row['ID_Trafo']),
                str(row['Tanggal_Uji']),
                row.get('H2'), row.get('CH4'), row.get('C2H6'), row.get('C2H4'), row.get('C2H2'),
                row.get('CO'), row.get('CO2'), row.get('O2'), row.get('N2'),
                row.get('BDV'), row.get('Acid'), row.get('Water'), row.get('IFT'),
                str(row['Status_Pemurnian']),
                str(row['Is_Anomali'])
            ))

        cursor.executemany("""
            INSERT INTO tabel_master (
                ID_Trafo, Tanggal_Uji, H2, CH4, C2H6, C2H4, C2H2,
                CO, CO2, O2, N2,
                BDV, Acid, Water, IFT, Status_Pemurnian, Is_Anomali
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()

    conn.close()

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM tabel_master ORDER BY Tanggal_Uji ASC", conn)
    conn.close()
    if not df.empty and 'Tanggal_Uji' in df.columns:
        df['Tanggal_Uji_DT'] = pd.to_datetime(df['Tanggal_Uji'])
    return df

def insert_data(data_dict):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tabel_master (
            ID_Trafo, Tanggal_Uji, H2, CH4, C2H6, C2H4, C2H2,
            CO, CO2, O2, N2,
            Acid, BDV, IFT, Water, Status_Pemurnian, Is_Anomali
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data_dict['ID_Trafo'],
        str(data_dict['Tanggal_Uji']),
        data_dict['H2'], data_dict['CH4'], data_dict['C2H6'], data_dict['C2H4'], data_dict['C2H2'],
        data_dict['CO'], data_dict['CO2'], data_dict['O2'], data_dict['N2'],
        data_dict['Acid'], data_dict['BDV'], data_dict['IFT'], data_dict['Water'],
        data_dict['Status_Pemurnian'], data_dict['Is_Anomali']
    ))
    conn.commit()
    conn.close()

def delete_data(record_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tabel_master WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def db_add_column(column_name, data_type="REAL"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    sanitized_col = "".join(c for c in column_name if c.isalnum() or c == "_")
    cursor.execute(f"ALTER TABLE tabel_master ADD COLUMN {sanitized_col} {data_type}")
    conn.commit()
    conn.close()

def db_drop_column(column_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    sanitized_col = "".join(c for c in column_name if c.isalnum() or c == "_")
    cursor.execute(f"ALTER TABLE tabel_master DROP COLUMN {sanitized_col}")
    conn.commit()
    conn.close()

init_db()

def check_anomaly(df_history, trafo_id, input_date, input_data):
    df_trafo = df_history[df_history['ID_Trafo'] == trafo_id].sort_values('Tanggal_Uji_DT')
    is_purification_valid = input_data['Status_Pemurnian'] in ["Reconditioning", "Reclaiming", "Oil Replacement", "Ganti Minyak"]
    is_new_trafo = df_trafo.empty
    
    exceeded = []
    if input_data['H2'] > 5000: exceeded.append(f"H2: {input_data['H2']} ppm (Limit: 5000 ppm)")
    if input_data['CH4'] > 3000: exceeded.append(f"CH4: {input_data['CH4']} ppm (Limit: 3000 ppm)")
    if input_data['C2H4'] > 2000: exceeded.append(f"C2H4: {input_data['C2H4']} ppm (Limit: 2000 ppm)")
    if input_data['C2H6'] > 1000: exceeded.append(f"C2H6: {input_data['C2H6']} ppm (Limit: 1000 ppm)")
    if input_data['C2H2'] > 500: exceeded.append(f"C2H2: {input_data['C2H2']} ppm (Limit: 500 ppm)")
    if input_data['CO'] > 5000: exceeded.append(f"CO: {input_data['CO']} ppm (Limit: 5000 ppm)")
    if input_data['CO2'] > 20000: exceeded.append(f"CO2: {input_data['CO2']} ppm (Limit: 20000 ppm)")
    if input_data['BDV'] > 100: exceeded.append(f"BDV: {input_data['BDV']} kV (Limit: 100 kV)")
    if input_data['Water'] > 100: exceeded.append(f"Water: {input_data['Water']} ppm (Limit: 100 ppm)")
    if input_data['Acid'] > 1.0: exceeded.append(f"Acid: {input_data['Acid']} mgKOH/g (Limit: 1.0 mgKOH/g)")
    
    if exceeded:
        return True, "Parameter values exceed physical laboratory thresholds:\n- " + "\n- ".join(exceeded)
    
    if input_data['C2H2'] > 0 and (input_data['H2'] == 0 or input_data['CH4'] == 0):
        return True, "DGA Physical Inconsistency: C2H2 detected without base gas H2 or CH4 generation."
    
    if is_purification_valid:
        if (input_data['H2'] > 50 or input_data['CH4'] > 50 or input_data['C2H4'] > 30 or input_data['C2H2'] > 2):
            return True, f"Gas concentrations are abnormally high for a post-purification condition ({input_data['Status_Pemurnian']})."
        return False, "Normal"
        
    if is_new_trafo:
        if (input_data['H2'] > 150 or input_data['CH4'] > 120 or input_data['C2H6'] > 65 or input_data['C2H4'] > 50 or input_data['C2H2'] > 2):
            return True, "Initial gas values for new transformer exceed IEEE baseline limits."
        return False, "Normal"

    rec_last = df_trafo.iloc[-1]
    last_date = rec_last['Tanggal_Uji_DT']
    days_diff = (input_date - last_date).days
    
    last_h2 = rec_last['H2'] if pd.notnull(rec_last['H2']) else 0
    last_ch4 = rec_last['CH4'] if pd.notnull(rec_last['CH4']) else 0
    
    if last_h2 == 0 and input_data['H2'] >= 100:
        return True, f"Extreme H2 spike from 0 ppm baseline to {input_data['H2']} ppm."
    if last_ch4 == 0 and input_data['CH4'] >= 80:
        return True, f"Extreme CH4 spike from 0 ppm baseline to {input_data['CH4']} ppm."
        
    if days_diff > 0:
        rate_h2 = ((input_data['H2'] - last_h2) / days_diff) * 30.43
        rate_ch4 = ((input_data['CH4'] - last_ch4) / days_diff) * 30.43
        rate_c2h4 = ((input_data['C2H4'] - (rec_last['C2H4'] if pd.notnull(rec_last['C2H4']) else 0)) / days_diff) * 30.43
        rate_c2h2 = ((input_data['C2H2'] - (rec_last['C2H2'] if pd.notnull(rec_last['C2H2']) else 0)) / days_diff) * 30.43
        
        rate_details = []
        if rate_h2 > 15: rate_details.append(f"H2 Rate: {rate_h2:.1f} ppm/month (Limit: 15 ppm/month)")
        if rate_ch4 > 12: rate_details.append(f"CH4 Rate: {rate_ch4:.1f} ppm/month (Limit: 12 ppm/month)")
        if rate_c2h4 > 5: rate_details.append(f"C2H4 Rate: {rate_c2h4:.1f} ppm/month (Limit: 5 ppm/month)")
        if rate_c2h2 > 0.2: rate_details.append(f"C2H2 Rate: {rate_c2h2:.1f} ppm/month (Limit: 0.2 ppm/month)")
        
        if rate_details:
            return True, "Monthly Gas Growth Rate exceeds acceptable thresholds:\n- " + "\n- ".join(rate_details)

    return False, "Normal"

def get_ieee_30year_thresholds(o2_n2_ratio):
    if o2_n2_ratio is not None and o2_n2_ratio <= 0.2:
        t1 = {'H2': 100, 'CH4': 110, 'C2H6': 150, 'C2H4': 90, 'C2H2': 1, 'CO': 900, 'CO2': 10000}
        t2 = {'H2': 200, 'CH4': 200, 'C2H6': 250, 'C2H4': 175, 'C2H2': 4, 'CO': 1100, 'CO2': 14000}
        t3_delta = {'H2': 40, 'CH4': 30, 'C2H6': 25, 'C2H4': 20, 'C2H2': 0.5, 'CO': 250, 'CO2': 2500}
        t4_rate = {'H2': 20, 'CH4': 10, 'C2H6': 9, 'C2H4': 7, 'C2H2': 0.5, 'CO': 100, 'CO2': 1000}
    else:
        t1 = {'H2': 40, 'CH4': 20, 'C2H6': 15, 'C2H4': 60, 'C2H2': 2, 'CO': 500, 'CO2': 5500}
        t2 = {'H2': 90, 'CH4': 30, 'C2H6': 40, 'C2H4': 125, 'C2H2': 7, 'CO': 600, 'CO2': 8000}
        t3_delta = {'H2': 25, 'CH4': 10, 'C2H6': 7, 'C2H4': 20, 'C2H2': 0.5, 'CO': 175, 'CO2': 1750}
        t4_rate = {'H2': 10, 'CH4': 3, 'C2H6': 2, 'C2H4': 5, 'C2H2': 0.5, 'CO': 80, 'CO2': 800}
    return t1, t2, t3_delta, t4_rate

def get_duval_minimum(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal"
    p_ch4 = (ch4 / total) * 100
    p_c2h4 = (c2h4 / total) * 100
    p_c2h2 = (c2h2 / total) * 100
    if p_c2h2 < 4:
        if p_c2h4 >= 38: return "T3"
        elif 23 <= p_c2h4 < 38: return "T2"
        return "T1"
    elif p_c2h2 >= 29 or (p_c2h2 >= 4 and p_c2h4 >= 23): return "D2"
    elif p_c2h2 >= 13 and p_c2h4 < 23: return "D1"
    return "T1"

def get_severity_score(label):
    s = str(label).upper()
    if "CRITICAL" in s: return 8
    if "D2" in s: return 7
    if "D1" in s: return 6
    if "T3" in s: return 5
    if "T2" in s: return 4
    if "T1" in s: return 3
    if "PD" in s: return 2
    if "CAUTION" in s: return 1
    return 0

def calculate_prognosis_and_prediction(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
        
    df = df_raw.copy()
    
    numeric_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'O2', 'N2', 'BDV', 'Acid', 'Water', 'IFT']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Status_Pemurnian' in df.columns:
        df['Status_Pemurnian'] = df['Status_Pemurnian'].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    else:
        df['Status_Pemurnian'] = np.nan

    df['Tanggal_Uji_DT'] = pd.to_datetime(df['Tanggal_Uji'], errors='coerce')

    forecast_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
    all_7_gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
    prediction_steps = 6
    final_results = []

    for trafo in df['ID_Trafo'].unique():
        df_trafo = df[df['ID_Trafo'] == trafo].copy()
        df_trafo['Tipe_Data'] = 'Historis'
        df_trafo = df_trafo.sort_values('Tanggal_Uji_DT').reset_index(drop=True)

        for gas in forecast_columns:
            df_trafo[f'GGR_{gas}'] = df_trafo[gas].diff().fillna(0)
            df_trafo[f'GGR_{gas}'] = df_trafo[f'GGR_{gas}'].apply(lambda x: round(max(x, 0), 2))

        last_time_dt = df_trafo['Tanggal_Uji_DT'].iloc[-1]
        last_data = df_trafo.iloc[-1].copy()

        two_years_ago = last_time_dt - relativedelta(years=2)
        df_temporal = df_trafo[df_trafo['Tanggal_Uji_DT'] >= two_years_ago].copy()

        purification_idx = df_temporal[df_temporal['Status_Pemurnian'].isin(['Oil Replacement', 'Reclaiming', 'Reconditioning', 'Ganti Minyak'])].index
        
        freeze_mode = False
        if not purification_idx.empty:
            last_purification_pos = purification_idx[-1]
            purification_date = df_temporal.loc[last_purification_pos, 'Tanggal_Uji_DT']
            df_new_era = df_temporal[df_temporal['Tanggal_Uji_DT'] >= purification_date].copy()
            
            new_era_age = (last_time_dt.year - purification_date.year) * 12 + (last_time_dt.month - purification_date.month)
            if len(df_new_era) < 6 or new_era_age < 4:
                freeze_mode = True
                data_train_arima = df_new_era.copy()
            else:
                data_train_arima = df_new_era.tail(6).copy()
        else:
            data_train_arima = df_temporal.tail(6).copy()

        future_predictions = {col: [] for col in forecast_columns}
        prediction_dates = [(last_time_dt + relativedelta(months=i+1)).strftime('%Y-%m-%d') for i in range(prediction_steps)]

        for col in forecast_columns:
            time_series = data_train_arima[col].values
            if freeze_mode or len(time_series) < 3:
                forecast = [time_series[-1] for _ in range(prediction_steps)]
            else:
                try:
                    model = ARIMA(time_series, order=(1, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=prediction_steps)
                except:
                    avg_diff = np.mean(np.diff(time_series[-3:])) if len(time_series) > 1 else 0
                    forecast = [time_series[-1] + (avg_diff * (i+1)) for i in range(prediction_steps)]
            
            forecast = np.maximum(forecast, time_series[-1])
            future_predictions[col] = forecast

        df_trafo_prediction_list = []
        for i in range(prediction_steps):
            new_row = last_data.copy()
            new_row['Tanggal_Uji'] = prediction_dates[i]
            new_row['Tanggal_Uji_DT'] = pd.to_datetime(prediction_dates[i])
            new_row['Tipe_Data'] = 'Prediksi'
            new_row['Status_Pemurnian'] = np.nan
            
            for col in forecast_columns:
                new_row[col] = round(future_predictions[col][i], 2)
                
            last_month_val = df_trafo_prediction_list[i-1] if i > 0 else last_data
            for gas in forecast_columns:
                ggr = new_row[gas] - (last_month_val[gas] if pd.notna(last_month_val[gas]) else 0)
                new_row[f'GGR_{gas}'] = round(max(ggr, 0), 2)
                
            df_trafo_prediction_list.append(new_row)

        df_trafo_prediction = pd.DataFrame(df_trafo_prediction_list)
        final_results.append(df_trafo)
        final_results.append(df_trafo_prediction)

    df_master = pd.concat(final_results, ignore_index=True)

    if os.path.exists(JALUR_MODEL_LOKAL):
        model_dga = joblib.load(JALUR_MODEL_LOKAL)
        vonis_ai_list = []
        for idx_rf, row_rf in df_master.iterrows():
            gas_vals = [row_rf['H2'], row_rf['CH4'], row_rf['C2H6'], row_rf['C2H4'], row_rf['C2H2']]
            if all(pd.notna(v) for v in gas_vals):
                try:
                    X_single = pd.DataFrame([gas_vals], columns=['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2'])
                    pred = model_dga.predict(X_single)[0]
                    vonis_ai_list.append(pred)
                except Exception:
                    vonis_ai_list.append('Normal')
            else:
                vonis_ai_list.append('Normal')
        df_master['Vonis_AI_Mentah'] = vonis_ai_list
    else:
        df_master['Vonis_AI_Mentah'] = 'Normal'

    dga_status_ieee_list = []
    status_dga_final_list = []
    status_paper_list = []
    recommendation_oa_list = []

    df_master['Tanggal_Uji_DT'] = pd.to_datetime(df_master['Tanggal_Uji'], errors='coerce')
    df_master = df_master.sort_values(['ID_Trafo', 'Tanggal_Uji_DT']).reset_index(drop=True)

    trafo_freeze_status = {}

    for idx, row in df_master.iterrows():
        trafo_id = row['ID_Trafo']

        status_p_str = str(row['Status_Pemurnian']).strip() if pd.notna(row['Status_Pemurnian']) else ""
        if status_p_str in ['Oil Replacement', 'Reclaiming', 'Reconditioning', 'Ganti Minyak']:
            trafo_freeze_status[trafo_id] = row['Tanggal_Uji_DT']

        is_currently_frozen = False
        if trafo_id in trafo_freeze_status:
            tgl_p = trafo_freeze_status[trafo_id]
            umb = (row['Tanggal_Uji_DT'].year - tgl_p.year) * 12 + (row['Tanggal_Uji_DT'].month - tgl_p.month)
            
            sub_df = df_master[
                (df_master['ID_Trafo'] == trafo_id) & 
                (df_master['Tanggal_Uji_DT'] >= tgl_p) & 
                (df_master['Tanggal_Uji_DT'] <= row['Tanggal_Uji_DT']) & 
                (df_master['Tipe_Data'] == 'Historis')
            ]
            if len(sub_df) < 6 or umb < 4:
                is_currently_frozen = True

        o2 = row['O2'] if pd.notna(row['O2']) else 0
        n2 = row['N2'] if pd.notna(row['N2']) else 0
        ratio_o2_n2 = o2 / (n2 + 1e-5) if n2 > 0 else None

        t1_limits, t2_limits, t3_delta_limits, t4_rate_limits = get_ieee_30year_thresholds(ratio_o2_n2)

        exceed_t1_any = any((pd.notna(row[g]) and row[g] > t1_limits[g]) for g in all_7_gases if g in row)
        exceed_t2_any = any((pd.notna(row[g]) and row[g] > t2_limits[g]) for g in all_7_gases if g in row)

        is_rates_anomali = False
        
        if not is_currently_frozen:
            if idx > 0 and df_master.loc[idx, 'ID_Trafo'] == df_master.loc[idx-1, 'ID_Trafo']:
                days_diff = (row['Tanggal_Uji_DT'] - df_master.loc[idx-1, 'Tanggal_Uji_DT']).days
                if days_diff > 0:
                    effective_days = max(days_diff, 3)
                    for g in all_7_gases:
                        if g in row and pd.notna(row[g]) and pd.notna(df_master.loc[idx-1, g]):
                            diff_ppm = row[g] - df_master.loc[idx-1, g]
                            annual_rate = (diff_ppm / effective_days) * 365.25

                            if g == 'C2H2' and diff_ppm >= 0.5: is_rates_anomali = True
                            elif g != 'C2H2' and diff_ppm > t3_delta_limits[g]: is_rates_anomali = True

                            if annual_rate > t4_rate_limits[g]: is_rates_anomali = True

        vonis_ai = row['Vonis_AI_Mentah']

        if not exceed_t1_any and not is_rates_anomali and not exceed_t2_any:
            status_ieee = "Status 1"
            status_dga_final = "Normal"
        elif exceed_t2_any or is_rates_anomali:
            status_ieee = "Status 3"
            ch4_v = row['CH4'] if pd.notna(row['CH4']) else 0
            c2h4_v = row['C2H4'] if pd.notna(row['C2H4']) else 0
            c2h2_v = row['C2H2'] if pd.notna(row['C2H2']) else 0
            
            vonis_fisika_pasti = get_duval_minimum(ch4_v, c2h4_v, c2h2_v)

            if c2h2_v == 0 and vonis_ai in ['D1', 'D2']:
                status_dga_final = vonis_fisika_pasti
            elif vonis_fisika_pasti == "T3" and vonis_ai in ['T1', 'T2', 'Normal', 'Caution']:
                status_dga_final = "T3"
            elif vonis_ai in ['Normal', 'Caution']:
                status_dga_final = vonis_fisika_pasti
            else:
                status_dga_final = vonis_ai
        else:
            status_ieee = "Status 2"
            status_dga_final = "Caution"

        dga_status_ieee_list.append(status_ieee)
        status_dga_final_list.append(status_dga_final)

        co = row['CO'] if 'CO' in row and pd.notna(row['CO']) else 0
        co2 = row['CO2'] if 'CO2' in row and pd.notna(row['CO2']) else 0
        ratio_co2_co = co2 / (co + 1e-5)

        if co > 1000 and ratio_co2_co < 3:
            if exceed_t1_any or is_rates_anomali:
                status_paper_list.append("Indication of Fault Involving Solid Paper Insulation")
            else:
                status_paper_list.append("Oil Oxidation (Restricted O2) - Paper Insulation Intact")
        elif co2 > 10000 and ratio_co2_co > 20:
            status_paper_list.append("Slow Degradation of Paper Insulation")
        else:
            status_paper_list.append("Normal")

        bdv = row['BDV'] if pd.notna(row['BDV']) else 0
        acid = row['Acid'] if pd.notna(row['Acid']) else 0
        water = row['Water'] if pd.notna(row['Water']) else 0
        ift = row['IFT'] if pd.notna(row['IFT']) else 0

        if is_currently_frozen:
            rec_oa = "New oil baseline detected. System under intensive post-maintenance monitoring (Freeze Mode Active)."
        elif acid > 0.30 or ift < 16:
            rec_oa = "OA RECOMMENDATION: MANDATORY TOTAL OIL REPLACEMENT (IEC 60422). Pressure Flushing Required."
        elif acid >= 0.15 or (20 <= ift <= 24):
            rec_oa = "OA RECOMMENDATION: Oil Reclaiming Required (Fuller's Earth)."
        elif bdv < 50 or water > 25:
            rec_oa = "OA RECOMMENDATION: Oil Reconditioning Required (Filtration & Vacuum Dehydration)."
        else:
            rec_oa = "Insulating oil physical conditions normal (IEC 60422)."

        recommendation_oa_list.append(rec_oa)

    df_master['DGA_Status_IEEE'] = dga_status_ieee_list
    df_master['Status_DGA'] = status_dga_final_list
    df_master['Paper_Status'] = status_paper_list
    df_master['OA_Recommendation'] = recommendation_oa_list

    df_master['Prognosis_DGA'] = ""
    df_master['Severity_Level'] = df_master['Status_DGA'].apply(get_severity_score)

    for trafo_id in df_master['ID_Trafo'].unique():
        idx_trafo = df_master[df_master['ID_Trafo'] == trafo_id].index
        list_idx = list(idx_trafo)

        for pos, idx in enumerate(list_idx):
            current_status = df_master.loc[idx, 'Status_DGA']
            current_severity = df_master.loc[idx, 'Severity_Level']
            row_date = df_master.loc[idx, 'Tanggal_Uji_DT']

            escalation_list = []
            max_severity_passed = current_severity
            recorded_status = set([current_status])

            for future_idx in list_idx[pos + 1:]:
                future_severity = df_master.loc[future_idx, 'Severity_Level']
                future_date = df_master.loc[future_idx, 'Tanggal_Uji_DT']
                future_status = df_master.loc[future_idx, 'Status_DGA']

                if pd.notna(future_date) and pd.notna(row_date):
                    days_dist = (future_date - row_date).days
                    month_diff = int(round(days_dist / 30.43))
                    if month_diff == 0 and days_dist > 15: month_diff = 1

                    if future_severity > max_severity_passed and future_status not in recorded_status:
                        escalation_list.append((future_status, month_diff))
                        max_severity_passed = future_severity
                        recorded_status.add(future_status)

            if current_severity == 0 or current_status == "Normal":
                if escalation_list:
                    parts = [f"Potential progression to {st} within {mo} month(s)" for st, mo in escalation_list]
                    conclusion = "Normal. " + " | ".join(parts)
                else:
                    conclusion = "Normal (Conditions predicted to remain stable)"
            else:
                if escalation_list:
                    parts = [f"potential escalation to {st} within {mo} month(s)" for st, mo in escalation_list]
                    conclusion = f"{current_status} detected | " + " | ".join(parts)
                else:
                    conclusion = f"{current_status} detected | Fault condition persisting"

            df_master.loc[idx, 'Prognosis_DGA'] = conclusion

    return df_master.drop(columns=['Tanggal_Uji_DT', 'Severity_Level', 'Vonis_AI_Mentah'], errors='ignore')

st.markdown("<div class='sap-header'>Transformer Digital Twin & DGA Diagnostic System</div>", unsafe_allow_html=True)

df_all = load_data()

tab_home, tab_input, tab_data, tab_insights, tab_trend = st.tabs([
    "Home", "Data Input", "Database & Editor", "Diagnostic Insights", "Trend Analysis"
])

with tab_home:
    st.markdown("<div class='sap-title'>System Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    This application is an enterprise **Digital Twin and Dissolved Gas Analysis (DGA) Diagnostic System** engineered for high-voltage power transformers operating beyond 30 years of operational life.
    
    The diagnostic engine combines physics-based domain standards with temporal machine learning:
    *   **IEEE C57.104-2019 Standard:** Evaluates absolute concentration limits and annual gas growth rates adapted for breathing vs sealed tank conditions.
    *   **Duval Triangle 1 Geometry:** Identifies exact fault types (T1, T2, T3 Thermal Faults or PD, D1, D2 Electrical Discharges).
    *   **Solid Paper Insulation Evaluation:** Assesses cellulose paper degradation through Carbon Monoxide and Carbon Dioxide gas generation ratios.
    *   **IEC 60422 Oil Analysis (OA):** Formulates oil health status and maintenance actions (Reconditioning, Reclaiming, or Full Oil Replacement) based on BDV, Acid, Water, and IFT physical parameters.
    *   **ARIMA Temporal Forecasting:** Generates 6-month predictive gas trajectories.
    *   **Random Forest Classifier:** Provides supervised machine learning fault pattern classification.
    
    Utilize the navigational tabs above to record laboratory observations, adjust database schema, review prognostic timelines, and visualize gas trends.
    """)

with tab_input:
    st.markdown("<div class='sap-title'>Laboratory Test Input</div>", unsafe_allow_html=True)
    
    existing_trafos = sorted(df_all['ID_Trafo'].unique().tolist()) if not df_all.empty else []
    trafo_options = existing_trafos + ["+ Add New Transformer..."]
    
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        selected_option = st.selectbox("Transformer ID", trafo_options)
        if selected_option == "+ Add New Transformer...":
            trafo_id_input = st.text_input("Enter New Transformer ID", placeholder="Example: Main_Transformer_06")
        else:
            trafo_id_input = selected_option

    with col_meta2:
        test_date = st.date_input("Test Date", datetime.now())

    with col_meta3:
        purification_status = st.selectbox("Oil Purification Status", ["Normal", "Reconditioning", "Reclaiming", "Oil Replacement"])

    with st.form("dga_input_form"):
        st.markdown("**Dissolved Gas Analysis (ppm):**")
        c1, c2, c3, c4 = st.columns(4)
        h2 = c1.number_input("H2", min_value=0.0, value=0.0, step=0.1)
        ch4 = c2.number_input("CH4", min_value=0.0, value=0.0, step=0.1)
        c2h6 = c3.number_input("C2H6", min_value=0.0, value=0.0, step=0.1)
        c2h4 = c4.number_input("C2H4", min_value=0.0, value=0.0, step=0.1)

        c5, c6, c7, c8 = st.columns(4)
        c2h2 = c5.number_input("C2H2", min_value=0.0, value=0.0, step=0.1)
        co = c6.number_input("CO", min_value=0.0, value=0.0, step=0.1)
        co2 = c7.number_input("CO2", min_value=0.0, value=0.0, step=0.1)
        o2 = c8.number_input("O2", min_value=0.0, value=0.0, step=0.1)

        c9, _, _, _ = st.columns(4)
        n2 = c9.number_input("N2", min_value=0.0, value=0.0, step=0.1)

        st.markdown("**Physical Oil Analysis (OA):**")
        o1, o2_col, o3, o4 = st.columns(4)
        bdv = o1.number_input("BDV (kV)", min_value=0.0, value=0.0, step=0.1)
        acid = o2_col.number_input("Acid Number (mgKOH/g)", min_value=0.0, value=0.0, step=0.01)
        water = o3.number_input("Water Content (ppm)", min_value=0.0, value=0.0, step=0.1)
        ift = o4.number_input("IFT (mN/m)", min_value=0.0, value=0.0, step=0.1)

        submit_btn = st.form_submit_button("Evaluate & Save Data")

    if submit_btn:
        if not trafo_id_input or trafo_id_input.strip() == "":
            st.error("Transformer ID cannot be empty. Please select or enter an ID.")
        else:
            input_dict = {
                'ID_Trafo': trafo_id_input.strip(),
                'Tanggal_Uji': test_date,
                'H2': h2, 'CH4': ch4, 'C2H6': c2h6, 'C2H4': c2h4, 'C2H2': c2h2,
                'CO': co, 'CO2': co2, 'O2': o2, 'N2': n2,
                'Acid': acid, 'BDV': bdv, 'IFT': ift, 'Water': water,
                'Status_Pemurnian': purification_status,
                'Is_Anomali': 'No'
            }
            
            has_anomaly, anomaly_reason = check_anomaly(df_all, trafo_id_input.strip(), pd.to_datetime(test_date), input_dict)
            
            if has_anomaly:
                st.session_state['pending_data'] = input_dict
                st.session_state['anomaly_reason'] = anomaly_reason
            else:
                insert_data(input_dict)
                st.success(f"Test data for Transformer '{trafo_id_input.strip()}' successfully saved.")
                st.rerun()

    if 'pending_data' in st.session_state and st.session_state['pending_data'] is not None:
        st.markdown("---")
        st.error(f"**INPUT ANOMALY DETECTED:**\n\n{st.session_state['anomaly_reason']}")
        
        st.info("""
        The parameter values or gas growth rates entered exceed normal limits. This data cannot be directly saved to the Master Table.

        **Before proceeding, please verify the following field conditions:**
        1. Were standard oil sampling procedures followed correctly?
        2. Is there potential for sample contamination (e.g., exposed or dirty sample bottles)?
        3. Were there external contaminants introduced during laboratory testing?
        """)
        
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("Force Save Data (Test Result Verified)"):
            p_data = st.session_state['pending_data']
            p_data['Is_Anomali'] = 'Yes'
            insert_data(p_data)
            st.session_state['pending_data'] = None
            st.session_state['anomaly_reason'] = None
            st.success("Anomalous data verified and successfully saved.")
            st.rerun()
            
        if col_c2.button("Cancel / Correct Input Data"):
            st.session_state['pending_data'] = None
            st.session_state['anomaly_reason'] = None
            st.info("Save operation cancelled. Please review the input parameters.")
            st.rerun()

with tab_data:
    st.markdown("<div class='sap-title'>Master Data Viewer & Schema Column Manager</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        with st.spinner("Processing analytical models and prognosis data..."):
            df_prognosis = calculate_prognosis_and_prediction(df_all)
            
        st.markdown("**View Display Table (Locked Cells)**")
        st.dataframe(df_prognosis, use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_prognosis.to_excel(writer, index=False, sheet_name='Data_Prognosis')
        
        st.download_button(
            label="Download Complete Prognosis Data (.XLSX)",
            data=buffer.getvalue(),
            file_name=f"Transformer_Prognosis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("---")
        st.markdown("<div class='sap-title'>Database Schema Column Editor (Add / Drop Columns)</div>", unsafe_allow_html=True)
        st.info("Note: Cell editing is restricted. Column schema modifications alter the underlying database structure upon confirmation.")
        
        col_mgmt1, col_mgmt2 = st.columns(2)
        
        with col_mgmt1:
            st.markdown("**Add Column to Database**")
            new_col_name = st.text_input("New Column Name", placeholder="Example: Notes").strip()
            new_col_type = st.selectbox("Data Type", ["REAL", "TEXT", "INTEGER"])
            
            if st.button("Request Add Column"):
                if new_col_name:
                    st.session_state['schema_action'] = ('ADD', new_col_name, new_col_type)
                else:
                    st.error("Column name cannot be blank.")
                    
        with col_mgmt2:
            st.markdown("**Drop Column from Database**")
            
            conn_schema = sqlite3.connect(DB_FILE)
            curr_cols = pd.read_sql_query("SELECT * FROM tabel_master LIMIT 1", conn_schema).columns.tolist()
            conn_schema.close()
            
            protected_cols = ['id', 'ID_Trafo', 'Tanggal_Uji']
            droppable_cols = [c for c in curr_cols if c not in protected_cols]
            
            col_to_drop = st.selectbox("Select Column to Drop", ["-- Select Column --"] + droppable_cols)
            
            if st.button("Request Drop Column"):
                if col_to_drop != "-- Select Column --":
                    st.session_state['schema_action'] = ('DROP', col_to_drop, None)
                else:
                    st.error("Please select a valid column.")

        if 'schema_action' in st.session_state and st.session_state['schema_action'] is not None:
            action, col_n, col_t = st.session_state['schema_action']
            st.markdown("---")
            st.warning(f"CONFIRMATION REQUIRED: Are you sure you want to {action} column '{col_n}' in the database?")
            
            c_conf1, col_conf2 = st.columns(2)
            if c_conf1.button("Confirm Schema Change"):
                try:
                    if action == 'ADD':
                        db_add_column(col_n, col_t)
                        st.success(f"Column '{col_n}' successfully added.")
                    elif action == 'DROP':
                        db_drop_column(col_n)
                        st.success(f"Column '{col_n}' successfully removed.")
                    st.session_state['schema_action'] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Schema execution failed: {str(e)}")
                    st.session_state['schema_action'] = None
                    
            if col_conf2.button("Cancel Schema Change"):
                st.session_state['schema_action'] = None
                st.rerun()

        st.markdown("---")
        st.markdown("**Row Record Deletion**")
        
        delete_options = df_all.apply(
            lambda x: f"ID: {x['id']} | Transformer: {x['ID_Trafo']} | Date: {x['Tanggal_Uji']}", axis=1
        ).tolist()
        
        record_to_delete = st.selectbox("Select Historical Record to Delete", ["-- Select Record --"] + delete_options)
        
        if st.button("Delete Selected Record"):
            if record_to_delete != "-- Select Record --":
                db_id = int(record_to_delete.split("|")[0].replace("ID:", "").strip())
                st.session_state['delete_confirm_id'] = db_id
            else:
                st.error("Please select a valid record.")
                
        if 'delete_confirm_id' in st.session_state and st.session_state['delete_confirm_id'] is not None:
            st.warning("Are you absolutely sure you want to delete this historical record?")
            col_del1, col_del2 = st.columns(2)
            if col_del1.button("Confirm Delete Record"):
                delete_data(st.session_state['delete_confirm_id'])
                st.session_state['delete_confirm_id'] = None
                st.success("Record deleted successfully.")
                st.rerun()
            if col_del2.button("Cancel Record Operation"):
                st.session_state['delete_confirm_id'] = None
                st.rerun()
    else:
        st.info("No data available in the database.")

with tab_insights:
    st.markdown("<div class='sap-title'>Diagnostic Insights & Timeline Tracking</div>", unsafe_allow_html=True)
    if not df_all.empty:
        df_prog_insight = calculate_prognosis_and_prediction(df_all)
        insight_trafo = st.selectbox("Select Transformer for Insights", df_prog_insight['ID_Trafo'].unique())
        
        df_insight_filtered = df_prog_insight[(df_prog_insight['ID_Trafo'] == insight_trafo) & (df_prog_insight['Tipe_Data'] == 'Historis')].sort_values('Tanggal_Uji').reset_index(drop=True)
        
        if not df_insight_filtered.empty:
            latest_record = df_insight_filtered.iloc[-1]
            
            fault_rows = df_insight_filtered[df_insight_filtered['Status_DGA'] != 'Normal']
            if not fault_rows.empty:
                first_fault_date = fault_rows.iloc[0]['Tanggal_Uji']
                first_fault_status = fault_rows.iloc[0]['Status_DGA']
                first_fault_text = f"{first_fault_date} (Initial State: {first_fault_status})"
            else:
                first_fault_text = "No Fault Ever Detected (Healthy Operation)"
            
            st.markdown(f"**Latest Evaluation Date:** {latest_record['Tanggal_Uji']}")
            
            st.markdown(f"<div class='insight-card'><div class='insight-label'>Current DGA Fault Status</div><div class='insight-value'>{latest_record['Status_DGA']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><div class='insight-label'>First DGA Fault Ever Detected On</div><div class='insight-value'>{first_fault_text}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><div class='insight-label'>Solid Paper Insulation Status</div><div class='insight-value'>{latest_record['Paper_Status']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><div class='insight-label'>Oil Maintenance Recommendation (IEC 60422)</div><div class='insight-value'>{latest_record['OA_Recommendation']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><div class='insight-label'>6-Month Prognosis & Escalation Timeline</div><div class='insight-value'>{latest_record['Prognosis_DGA']}</div></div>", unsafe_allow_html=True)
        else:
            st.info("Insufficient historical data to generate insights for this transformer.")
    else:
        st.info("No data available.")

with tab_trend:
    st.markdown("<div class='sap-title'>Trend Analysis & Oil Quality Gauges</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        trafo_filter = st.selectbox("Select Transformer", df_all['ID_Trafo'].unique(), key="trend_selector")
        
        df_graph = calculate_prognosis_and_prediction(df_all)
        df_filtered = df_graph[df_graph['ID_Trafo'] == trafo_filter].sort_values('Tanggal_Uji').reset_index(drop=True)
        
        fig_gas = go.Figure()
        
        gas_list = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
        colors = ['#2563EB', '#D97706', '#059669', '#DC2626', '#7C3AED']
        
        df_hist_only = df_filtered[df_filtered['Tipe_Data'] == 'Historis']
        last_hist_date = df_hist_only['Tanggal_Uji'].iloc[-1] if not df_hist_only.empty else None

        for idx_g, gas in enumerate(gas_list):
            if gas in df_filtered.columns:
                fig_gas.add_trace(go.Scatter(
                    x=df_filtered['Tanggal_Uji'],
                    y=df_filtered[gas],
                    mode='lines+markers',
                    name=gas,
                    line=dict(color=colors[idx_g % len(colors)], width=2),
                    marker=dict(size=6)
                ))

        if last_hist_date:
            fig_gas.add_vline(
                x=last_hist_date,
                line_width=2,
                line_dash="dash",
                line_color="#2563EB"
            )
            fig_gas.add_annotation(
                x=last_hist_date,
                y=1.02,
                yref="paper",
                text=f"Current Baseline ({last_hist_date})",
                showarrow=False,
                font=dict(color="#2563EB", size=12, family="Inter")
            )

        fig_gas.update_layout(
            title=f"DGA Gas Concentration Evolution & Forecast - {trafo_filter}",
            xaxis_title="Test Date / Prediction",
            yaxis_title="Concentration (ppm)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig_gas, use_container_width=True)

        st.markdown("<div class='sap-title'>Current Oil Analysis Physical Metrics</div>", unsafe_allow_html=True)
        
        if not df_hist_only.empty:
            last_oa = df_hist_only.iloc[-1]
            
            g1, g2, g3, g4 = st.columns(4)
            
            bdv_val = last_oa['BDV'] if pd.notna(last_oa['BDV']) else 0
            fig_bdv = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bdv_val,
                title={'text': "BDV (kV) [Min 50]"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FEF2F2"},
                        {'range': [30, 50], 'color': "#FEF9C3"},
                        {'range': [50, 100], 'color': "#E0F2FE"}
                    ]
                }
            ))
            fig_bdv.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            g1.plotly_chart(fig_bdv, use_container_width=True)

            acid_val = last_oa['Acid'] if pd.notna(last_oa['Acid']) else 0
            fig_acid = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acid_val,
                title={'text': "Acid (mgKOH/g) [Max 0.15]"},
                gauge={
                    'axis': {'range': [0, 0.5]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 0.15], 'color': "#E0F2FE"},
                        {'range': [0.15, 0.30], 'color': "#FEF9C3"},
                        {'range': [0.30, 0.5], 'color': "#FEF2F2"}
                    ]
                }
            ))
            fig_acid.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            g2.plotly_chart(fig_acid, use_container_width=True)

            water_val = last_oa['Water'] if pd.notna(last_oa['Water']) else 0
            fig_water = go.Figure(go.Indicator(
                mode="gauge+number",
                value=water_val,
                title={'text': "Water (ppm) [Max 25]"},
                gauge={
                    'axis': {'range': [0, 60]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 25], 'color': "#E0F2FE"},
                        {'range': [25, 40], 'color': "#FEF9C3"},
                        {'range': [40, 60], 'color': "#FEF2F2"}
                    ]
                }
            ))
            fig_water.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            g3.plotly_chart(fig_water, use_container_width=True)

            ift_val = last_oa['IFT'] if pd.notna(last_oa['IFT']) else 0
            fig_ift = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ift_val,
                title={'text': "IFT (mN/m) [Min 24]"},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 16], 'color': "#FEF2F2"},
                        {'range': [16, 24], 'color': "#FEF9C3"},
                        {'range': [24, 50], 'color': "#E0F2FE"}
                    ]
                }
            ))
            fig_ift.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            g4.plotly_chart(fig_ift, use_container_width=True)
    else:
        st.info("No data available.")
