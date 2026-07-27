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

# STYLING SCADA / HMI INDUSTRIAL PANEL (IBM PLEX MONO & DARK SUBSTATION TONE)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

    * {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #121926;
        color: #D1D5DB;
    }
    
    .main { 
        background-color: #121926; 
    }
    
    code, pre, .stDataFrame, input, button, select, div[data-baseweb="select"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .stButton>button {
        background-color: #243044;
        color: #E5E7EB;
        border-radius: 0px;
        border: 1px solid #374151;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        width: 100%;
        transition: none;
    }
    
    .stButton>button:hover {
        background-color: #374151;
        color: #FFFFFF;
        border: 1px solid #4B5563;
    }
    
    .sap-header {
        background-color: #1A2332;
        padding: 16px 20px;
        color: #F3F4F6;
        font-size: 18px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        border-bottom: 3px solid #D97706;
        border-top: 1px solid #2A364F;
        border-left: 1px solid #2A364F;
        border-right: 1px solid #2A364F;
        margin-bottom: 20px;
    }
    
    .sap-title {
        background-color: #1A2332;
        padding: 10px 14px;
        color: #F3F4F6;
        font-size: 14px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        border-left: 4px solid #0D9488;
        border-top: 1px solid #2A364F;
        border-right: 1px solid #2A364F;
        border-bottom: 1px solid #2A364F;
        margin-bottom: 16px;
    }
    
    .hmi-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1A2332;
        border: 1px solid #2A364F;
        margin-bottom: 20px;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .hmi-table td {
        padding: 10px 14px;
        border: 1px solid #2A364F;
        font-size: 13px;
    }
    
    .hmi-label {
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 11px;
        width: 30%;
    }
    
    .hmi-value {
        color: #F3F4F6;
        font-weight: 600;
        width: 20%;
    }

    div[data-baseweb="tab-list"] {
        background-color: #1A2332;
        border: 1px solid #2A364F;
        padding: 2px;
    }

    button[data-baseweb="tab"] {
        border-radius: 0px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        color: #9CA3AF !important;
    }

    button[aria-selected="true"] {
        background-color: #243044 !important;
        color: #F3F4F6 !important;
        border-bottom: 2px solid #D97706 !important;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "dga_database.db"
CSV_SEED_FILE = "Data_Uji_Trafo.csv"
CSV_NAMEPLATE_FILE = "Nameplate_Trafo.csv"
JALUR_MODEL_LOKAL = 'model_dga_7classes_v2.pkl'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_trafo_metadata (
            ID_Trafo TEXT PRIMARY KEY,
            Manufacturer TEXT,
            Serial_Number TEXT,
            Year_Manufactured INTEGER,
            Model_Type TEXT,
            Capacity_MVA REAL,
            Phase_Count INTEGER,
            Nominal_Voltage_kV TEXT,
            Nominal_Current_A TEXT,
            Frequency_Hz REAL,
            Vector_Group TEXT,
            Impedance_Pct REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Trafo TEXT NOT NULL,
            Tanggal_Uji TEXT NOT NULL,
            H2 REAL, CH4 REAL, C2H6 REAL, C2H4 REAL, C2H2 REAL,
            CO REAL, CO2 REAL, Ratio_O2_N2 REAL,
            BDV REAL, Acid REAL, Water REAL, IFT REAL,
            DDF REAL, Resistivity REAL, Colour_ISO2049 REAL, Sediment_Sludge TEXT,
            Corrosive_Sulphur TEXT, Particles_ISO TEXT, Inhibitor_Content REAL,
            Passivator_Content REAL, Flash_Point REAL, PCB_Content REAL,
            Status_Pemurnian TEXT,
            Is_Anomali TEXT DEFAULT 'No'
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM tabel_trafo_metadata")
    if cursor.fetchone()[0] == 0 and os.path.exists(CSV_NAMEPLATE_FILE):
        df_meta_csv = pd.read_csv(CSV_NAMEPLATE_FILE)
        meta_records = []
        for _, row in df_meta_csv.iterrows():
            meta_records.append((
                str(row['ID_Trafo']), str(row['Manufacturer']), str(row['Serial_Number']),
                int(row['Year_Manufactured']), str(row['Model_Type']), float(row['Capacity_MVA']),
                int(row['Phase_Count']), str(row['Nominal_Voltage_kV']), str(row['Nominal_Current_A']),
                float(row['Frequency_Hz']), str(row['Vector_Group']), float(row['Impedance_Pct'])
            ))
            
        cursor.executemany("""
            INSERT OR IGNORE INTO tabel_trafo_metadata (
                ID_Trafo, Manufacturer, Serial_Number, Year_Manufactured, Model_Type,
                Capacity_MVA, Phase_Count, Nominal_Voltage_kV, Nominal_Current_A,
                Frequency_Hz, Vector_Group, Impedance_Pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, meta_records)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM tabel_master")
    if cursor.fetchone()[0] == 0 and os.path.exists(CSV_SEED_FILE):
        df_csv = pd.read_csv(CSV_SEED_FILE)
        df_csv['Tanggal_Uji'] = pd.to_datetime(df_csv['Tanggal_Uji']).dt.strftime('%Y-%m-%d')
        df_csv['Status_Pemurnian'] = df_csv['Status_Pemurnian'].fillna('Normal')
        df_csv['Is_Anomali'] = 'No'

        records = []
        for _, row in df_csv.iterrows():
            records.append((
                str(row['ID_Trafo']),
                str(row['Tanggal_Uji']),
                float(row.get('H2', 0.0)), float(row.get('CH4', 0.0)), float(row.get('C2H6', 0.0)), float(row.get('C2H4', 0.0)), float(row.get('C2H2', 0.0)),
                float(row.get('CO', 0.0)), float(row.get('CO2', 0.0)), float(row.get('Ratio_O2_N2', 0.32)),
                float(row.get('BDV', 50.0)), float(row.get('Acid', 0.05)), float(row.get('Water', 12.0)), float(row.get('IFT', 30.0)),
                float(row.get('DDF', 0.008)), float(row.get('Resistivity', 70.0)), float(row.get('Colour_ISO2049', 1.5)), str(row.get('Sediment_Sludge', 'No')),
                str(row.get('Corrosive_Sulphur', 'Non-Corrosive')), str(row.get('Particles_ISO', 'Good')), float(row.get('Inhibitor_Content', 80.0)),
                float(row.get('Passivator_Content', 100.0)), float(row.get('Flash_Point', 145.0)), float(row.get('PCB_Content', 0.0)),
                str(row['Status_Pemurnian']), str(row['Is_Anomali'])
            ))

        cursor.executemany("""
            INSERT INTO tabel_master (
                ID_Trafo, Tanggal_Uji, H2, CH4, C2H6, C2H4, C2H2, CO, CO2, Ratio_O2_N2,
                BDV, Acid, Water, IFT, DDF, Resistivity, Colour_ISO2049, Sediment_Sludge,
                Corrosive_Sulphur, Particles_ISO, Inhibitor_Content, Passivator_Content,
                Flash_Point, PCB_Content, Status_Pemurnian, Is_Anomali
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

def load_metadata():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM tabel_trafo_metadata ORDER BY ID_Trafo ASC", conn)
    conn.close()
    return df

def insert_metadata(meta_dict):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO tabel_trafo_metadata (
            ID_Trafo, Manufacturer, Serial_Number, Year_Manufactured, Model_Type,
            Capacity_MVA, Phase_Count, Nominal_Voltage_kV, Nominal_Current_A,
            Frequency_Hz, Vector_Group, Impedance_Pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta_dict['ID_Trafo'], meta_dict['Manufacturer'], meta_dict['Serial_Number'],
        meta_dict['Year_Manufactured'], meta_dict['Model_Type'], meta_dict['Capacity_MVA'],
        meta_dict['Phase_Count'], meta_dict['Nominal_Voltage_kV'], meta_dict['Nominal_Current_A'],
        meta_dict['Frequency_Hz'], meta_dict['Vector_Group'], meta_dict['Impedance_Pct']
    ))
    conn.commit()
    conn.close()

def insert_data(data_dict):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tabel_master (
            ID_Trafo, Tanggal_Uji, H2, CH4, C2H6, C2H4, C2H2, CO, CO2, Ratio_O2_N2,
            BDV, Acid, Water, IFT, DDF, Resistivity, Colour_ISO2049, Sediment_Sludge,
            Corrosive_Sulphur, Particles_ISO, Inhibitor_Content, Passivator_Content,
            Flash_Point, PCB_Content, Status_Pemurnian, Is_Anomali
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data_dict['ID_Trafo'], str(data_dict['Tanggal_Uji']),
        data_dict['H2'], data_dict['CH4'], data_dict['C2H6'], data_dict['C2H4'], data_dict['C2H2'],
        data_dict['CO'], data_dict['CO2'], data_dict['Ratio_O2_N2'],
        data_dict['BDV'], data_dict['Acid'], data_dict['Water'], data_dict['IFT'],
        data_dict['DDF'], data_dict['Resistivity'], data_dict['Colour_ISO2049'], data_dict['Sediment_Sludge'],
        data_dict['Corrosive_Sulphur'], data_dict['Particles_ISO'], data_dict['Inhibitor_Content'],
        data_dict['Passivator_Content'], data_dict['Flash_Point'], data_dict['PCB_Content'],
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
    if input_data['Colour_ISO2049'] > 8.0: exceeded.append(f"Colour: {input_data['Colour_ISO2049']} ISO 2049 (Max Scale: 8.0)")
    
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

def get_ieee_thresholds(o2_n2_ratio, age_years, period_months=12):
    is_sealed = (o2_n2_ratio is not None and o2_n2_ratio <= 0.2)
    
    if age_years is None or age_years <= 0:
        age_cat = 'unknown'
    elif 1 <= age_years <= 9:
        age_cat = '1_9'
    elif 10 <= age_years <= 30:
        age_cat = '10_30'
    else:
        age_cat = 'gt_30'

    if is_sealed:
        t1_map = {
            'unknown': {'H2': 80,  'CH4': 90,  'C2H6': 30,  'C2H4': 20, 'C2H2': 1, 'CO': 900, 'CO2': 9000},
            '1_9':     {'H2': 75,  'CH4': 45,  'C2H6': 30,  'C2H4': 20, 'C2H2': 1, 'CO': 900, 'CO2': 5000},
            '10_30':   {'H2': 80,  'CH4': 90,  'C2H6': 90,  'C2H4': 50, 'C2H2': 1, 'CO': 900, 'CO2': 10000},
            'gt_30':   {'H2': 100, 'CH4': 110, 'C2H6': 150, 'C2H4': 90, 'C2H2': 1, 'CO': 900, 'CO2': 10000}
        }
        t2_map = {
            'unknown': {'H2': 200, 'CH4': 150, 'C2H6': 175, 'C2H4': 100, 'C2H2': 2, 'CO': 1100, 'CO2': 12500},
            '1_9':     {'H2': 200, 'CH4': 100, 'C2H6': 70,  'C2H4': 40,  'C2H2': 2, 'CO': 1100, 'CO2': 7000},
            '10_30':   {'H2': 200, 'CH4': 150, 'C2H6': 175, 'C2H4': 95,  'C2H2': 2, 'CO': 1100, 'CO2': 14000},
            'gt_30':   {'H2': 200, 'CH4': 200, 'C2H6': 250, 'C2H4': 175, 'C2H2': 4, 'CO': 1100, 'CO2': 14000}
        }
        t3_delta = {'H2': 40, 'CH4': 30, 'C2H6': 25, 'C2H4': 20, 'C2H2': 0.5, 'CO': 250, 'CO2': 2500}
        
        if period_months < 10:
            t4_rate = {'H2': 25, 'CH4': 10, 'C2H6': 15, 'C2H4': 10, 'C2H2': 0.01, 'CO': 100, 'CO2': 1750}
        else:
            t4_rate = {'H2': 10, 'CH4': 4,  'C2H6': 9,  'C2H4': 5,  'C2H2': 0.01, 'CO': 100, 'CO2': 1000}

    else:
        t1_map = {
            'unknown': {'H2': 40, 'CH4': 20, 'C2H6': 15, 'C2H4': 50, 'C2H2': 2, 'CO': 500, 'CO2': 5000},
            '1_9':     {'H2': 40, 'CH4': 20, 'C2H6': 15, 'C2H4': 25, 'C2H2': 2, 'CO': 500, 'CO2': 3500},
            '10_30':   {'H2': 40, 'CH4': 20, 'C2H6': 15, 'C2H4': 50, 'C2H2': 2, 'CO': 500, 'CO2': 5000},
            'gt_30':   {'H2': 40, 'CH4': 20, 'C2H6': 15, 'C2H4': 60, 'C2H2': 2, 'CO': 500, 'CO2': 5500}
        }
        t2_map = {
            'unknown': {'H2': 90, 'CH4': 50, 'C2H6': 40, 'C2H4': 100, 'C2H2': 7, 'CO': 600, 'CO2': 7000},
            '1_9':     {'H2': 90, 'CH4': 60, 'C2H6': 30, 'C2H4': 80,  'C2H2': 7, 'CO': 600, 'CO2': 5000},
            '10_30':   {'H2': 90, 'CH4': 60, 'C2H6': 40, 'C2H4': 125, 'C2H2': 7, 'CO': 600, 'CO2': 8000},
            'gt_30':   {'H2': 90, 'CH4': 30, 'C2H6': 40, 'C2H4': 125, 'C2H2': 7, 'CO': 600, 'CO2': 8000}
        }
        t3_delta = {'H2': 25, 'CH4': 10, 'C2H6': 7, 'C2H4': 20, 'C2H2': 0.5, 'CO': 175, 'CO2': 1750}
        
        if period_months < 10:
            t4_rate = {'H2': 50, 'CH4': 15, 'C2H6': 2, 'C2H4': 7, 'C2H2': 0.01, 'CO': 200, 'CO2': 1000}
        else:
            t4_rate = {'H2': 20, 'CH4': 3,  'C2H6': 3, 'C2H4': 7, 'C2H2': 0.01, 'CO': 80,  'CO2': 800}

    return t1_map[age_cat], t2_map[age_cat], t3_delta, t4_rate

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
    df_meta = load_metadata()
    
    numeric_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'Ratio_O2_N2', 'BDV', 'Acid', 'Water', 'IFT', 'DDF', 'Resistivity', 'Colour_ISO2049', 'Inhibitor_Content', 'Passivator_Content', 'Flash_Point', 'PCB_Content']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

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
                val_last = time_series[-1] if len(time_series) > 0 else 0.0
                forecast = [val_last for _ in range(prediction_steps)]
            else:
                try:
                    model = ARIMA(time_series, order=(1, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=prediction_steps)
                except:
                    avg_diff = np.mean(np.diff(time_series[-3:])) if len(time_series) > 1 else 0
                    forecast = [time_series[-1] + (avg_diff * (i+1)) for i in range(prediction_steps)]
            
            forecast = np.maximum(forecast, time_series[-1] if len(time_series) > 0 else 0.0)
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
                prev_v = last_month_val[gas] if pd.notna(last_month_val[gas]) else 0
                ggr = new_row[gas] - prev_v
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
    recommendation_oa_reason_list = []

    df_master['Tanggal_Uji_DT'] = pd.to_datetime(df_master['Tanggal_Uji'], errors='coerce')
    df_master = df_master.sort_values(['ID_Trafo', 'Tanggal_Uji_DT']).reset_index(drop=True)

    trafo_freeze_status = {}

    for idx, row in df_master.iterrows():
        trafo_id = row['ID_Trafo']

        meta_match = df_meta[df_meta['ID_Trafo'] == trafo_id]
        if not meta_match.empty and pd.notna(meta_match.iloc[0]['Year_Manufactured']):
            year_manuf = meta_match.iloc[0]['Year_Manufactured']
            calculated_age = row['Tanggal_Uji_DT'].year - year_manuf
        else:
            calculated_age = None

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

        o2_n2_ratio = row.get('Ratio_O2_N2', 0.32)

        t1_limits, t2_limits, t3_delta_limits, t4_rate_limits = get_ieee_thresholds(o2_n2_ratio, calculated_age)

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

        # LOGIKA REKOMENDASI PEMURNIAN MINYAK (IEC 60422 KATEGORI C < 72.5 kV)
        bdv = float(row.get('BDV', 0.0))
        acid = float(row.get('Acid', 0.0))
        water = float(row.get('Water', 0.0))
        ift = float(row.get('IFT', 0.0))
        ddf = float(row.get('DDF', 0.0))
        resistivity = float(row.get('Resistivity', 0.0))
        colour = float(row.get('Colour_ISO2049', 1.0))
        sediment_sludge = str(row.get('Sediment_Sludge')).strip()
        corrosive_sulfur = str(row.get('Corrosive_Sulphur')).strip()

        reasons = []

        if is_currently_frozen:
            rec_oa = "OA STATUS: FREEZE MODE ACTIVE"
            reason_str = "New oil baseline detected. System under intensive post-maintenance monitoring."
        else:
            # 1. Mandatory Total Replacement (IEC 60422 Category C limits)
            if acid > 0.30:
                reasons.append(f"Acid Number critical ({acid:.2f} mgKOH/g > 0.30 Poor limit)")
            if 0 < ift < 22:
                reasons.append(f"Interfacial Tension critical ({ift:.1f} mN/m < 22 Poor limit)")
            if colour >= 6.0:
                reasons.append(f"Colour Scale critical ({colour:.1f} ISO 2049 >= 6.0 limit)")
            if acid > 0.20 and ddf > 0.50 and 0 < resistivity < 4:
                reasons.append(f"Severe Combined Aging (Acid: {acid:.2f}, DDF: {ddf:.3f}, Resistivity: {resistivity:.1f})")

            if reasons:
                rec_oa = "OA RECOMMENDATION: MANDATORY TOTAL OIL REPLACEMENT (IEC 60422)"
                reason_str = " | ".join(reasons) + " -> Pressure Flushing & Full Oil Replacement Required."
            elif corrosive_sulfur == "Corrosive":
                rec_oa = "OA RECOMMENDATION: PASSIVATION OR RECLAIMING REQUIRED (IEC 60422)"
                reason_str = "Corrosive Sulphur detected -> Risk of copper sulfide deposition. Perform Passivation or Reclaiming."
            else:
                # 2. Reclaiming (Fuller's Earth Adsorption)
                reclaim_reasons = []
                if acid >= 0.15:
                    reclaim_reasons.append(f"Acid elevated ({acid:.2f} mgKOH/g >= 0.15 Fair limit)")
                if 22 <= ift <= 28:
                    reclaim_reasons.append(f"IFT degraded ({ift:.1f} mN/m within 22-28 Fair range)")
                if ddf > 0.10:
                    reclaim_reasons.append(f"DDF/Tan Delta high ({ddf:.3f} > 0.10 Fair limit)")
                if 0 < resistivity < 60:
                    reclaim_reasons.append(f"Resistivity low ({resistivity:.1f} Gohm.m < 60 Fair limit)")
                if colour >= 4.0:
                    reclaim_reasons.append(f"Oil Colour degraded ({colour:.1f} ISO 2049 >= 4.0 Poor limit)")
                if sediment_sludge == "Sludge":
                    reclaim_reasons.append("Precipitable Sludge detected")

                if reclaim_reasons:
                    rec_oa = "OA RECOMMENDATION: OIL RECLAIMING REQUIRED (FULLER'S EARTH)"
                    reason_str = " | ".join(reclaim_reasons) + " -> Fuller's Earth Adsorption & Re-inhibition needed."
                else:
                    # 3. Reconditioning (Filtration & Dehydration)
                    recond_reasons = []
                    if 0 < bdv < 40:
                        recond_reasons.append(f"Breakdown Voltage low ({bdv:.1f} kV < 40 Fair limit)")
                    if water > 30:
                        recond_reasons.append(f"Water Content high ({water:.1f} ppm > 30 Fair limit)")

                    if recond_reasons:
                        rec_oa = "OA RECOMMENDATION: OIL RECONDITIONING REQUIRED (FILTRATION)"
                        reason_str = " | ".join(recond_reasons) + " -> Vacuum Dehydration & Mechanical Filtration needed."
                    else:
                        rec_oa = "OA STATUS: NORMAL OPERATIONAL CONDITION (IEC 60422)"
                        reason_str = "All active oil physical and chemical parameters are within normal limits (Category C)."

        recommendation_oa_list.append(rec_oa)
        recommendation_oa_reason_list.append(reason_str)

    df_master['DGA_Status_IEEE'] = dga_status_ieee_list
    df_master['Status_DGA'] = status_dga_final_list
    df_master['Paper_Status'] = status_paper_list
    df_master['OA_Recommendation'] = recommendation_oa_list
    df_master['OA_Recommendation_Reason'] = recommendation_oa_reason_list

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

st.markdown("<div class='sap-header'>TRANSFORMER SUBSTATION MONITORING SYSTEM</div>", unsafe_allow_html=True)

df_all = load_data()
df_metadata = load_metadata()

tab_home, tab_nameplate, tab_input, tab_data, tab_insights, tab_trend = st.tabs([
    "System Overview", "Transformer Identity & Nameplate", "Data Input Form", "Database Master & Ledger", "Diagnostic Insights", "HMI Trend Analysis"
])

with tab_home:
    st.markdown("<div class='sap-title'>SYSTEM ARCHITECTURE & TECHNICAL SPECIFICATIONS</div>", unsafe_allow_html=True)
    st.markdown("""
    This platform operates as an enterprise **HMI/SCADA Digital Twin and Dissolved Gas Analysis (DGA) Diagnostic Engine** engineered for high-voltage power transformers across all age categories.
    
    The diagnostic engine combines physics-based domain standards with temporal machine learning:
    *   **Dynamic IEEE C57.104-2019 Standard:** Evaluates gas limits based on dynamically calculated operating age (Test Date - Year Manufactured) and O2/N2 ratio.
    *   **Duval Triangle 1 Geometry:** Identifies exact fault types (T1, T2, T3 Thermal Faults or PD, D1, D2 Electrical Discharges).
    *   **Solid Paper Insulation Evaluation:** Assesses cellulose paper degradation through Carbon Monoxide and Carbon Dioxide gas generation ratios.
    *   **Expanded IEC 60422 Oil Analysis (OA) Category C (< 72.5 kV):** Formulates oil health status and maintenance actions based on active decision parameters (BDV, Water, Acid, IFT, DDF, Resistivity, ISO 2049 Colour, Corrosive Sulphur, and Sludge).
    *   **ARIMA Temporal Forecasting:** Generates 6-month predictive gas trajectories.
    *   **Random Forest Classifier:** Provides supervised machine learning fault pattern classification.
    
    Utilize the navigational tabs above to register transformer nameplates, record laboratory observations, manage historical database records, review prognostic timelines, and visualize gas trends.
    """)

# TAB 2: NAMEPLATE REGISTRATION PANEL
with tab_nameplate:
    st.markdown("<div class='sap-title'>TRANSFORMER NAMEPLATE REGISTRATION PANEL</div>", unsafe_allow_html=True)
    
    st.markdown("**Registered Transformer Nameplates**")
    st.dataframe(df_metadata, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Register / Update Unit Nameplate**")
    
    with st.form("nameplate_form"):
        col_n1, col_n2, col_n3 = st.columns(3)
        reg_id = col_n1.text_input("Transformer ID", placeholder="Example: Main_Transformer_06").strip()
        reg_manuf = col_n2.text_input("Manufacturer Name", placeholder="Example: ABB Power")
        reg_sn = col_n3.text_input("Serial Number", placeholder="Example: SN-2024-99")

        col_n4, col_n5, col_n6 = st.columns(3)
        reg_year = col_n4.number_input("Year Manufactured", min_value=1950, max_value=2026, value=2010, step=1)
        reg_model = col_n5.text_input("Model / Type Designation", placeholder="Example: ONAN-TR6")
        reg_mva = col_n6.number_input("Power Capacity (MVA)", min_value=0.1, value=30.0, step=0.5)

        col_n7, col_n8, col_n9 = st.columns(3)
        reg_phase = col_n7.selectbox("Number of Phases", [3, 1])
        reg_voltage = col_n8.text_input("Nominal Voltage (kV)", placeholder="Example: 150/20")
        reg_current = col_n9.text_input("Nominal Current (A)", placeholder="Example: 115/866")

        col_n10, col_n11, col_n12 = st.columns(3)
        reg_freq = col_n10.number_input("Frequency (Hz)", min_value=40.0, max_value=70.0, value=50.0, step=1.0)
        reg_vector = col_n11.text_input("Vector Group", placeholder="Example: YNd11")
        reg_impedance = col_n12.number_input("Short Circuit Impedance (%)", min_value=0.1, value=12.5, step=0.1)

        btn_reg = st.form_submit_button("REGISTER NAMEPLATE SPECIFICATIONS")

    if btn_reg:
        if not reg_id:
            st.error("Transformer ID cannot be empty.")
        else:
            meta_dict = {
                'ID_Trafo': reg_id,
                'Manufacturer': reg_manuf,
                'Serial_Number': reg_sn,
                'Year_Manufactured': int(reg_year),
                'Model_Type': reg_model,
                'Capacity_MVA': float(reg_mva),
                'Phase_Count': int(reg_phase),
                'Nominal_Voltage_kV': reg_voltage,
                'Nominal_Current_A': reg_current,
                'Frequency_Hz': float(reg_freq),
                'Vector_Group': reg_vector,
                'Impedance_Pct': float(reg_impedance)
            }
            insert_metadata(meta_dict)
            st.success(f"Nameplate specification for '{reg_id}' successfully registered.")
            st.rerun()

# TAB 3: DATA INPUT FORM
with tab_input:
    st.markdown("<div class='sap-title'>LABORATORY OBSERVATION INPUT PANEL</div>", unsafe_allow_html=True)
    
    registered_trafos = sorted(df_metadata['ID_Trafo'].unique().tolist()) if not df_metadata.empty else []
    
    if not registered_trafos:
        st.warning("No transformers registered. Please register unit nameplate first in Tab [Transformer Identity & Nameplate].")
    else:
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            trafo_id_input = st.selectbox("SELECT REGISTERED TRANSFORMER", registered_trafos)

        with col_meta2:
            test_date = st.date_input("TEST DATE", datetime.now())

        with col_meta3:
            purification_status = st.selectbox("OIL PURIFICATION STATUS", ["Normal", "Reconditioning", "Reclaiming", "Oil Replacement"])

        with st.form("dga_input_form"):
            st.markdown("**DISSOLVED GAS ANALYSIS (PPM) & O2/N2 RATIO:**")
            c1, c2, c3, c4 = st.columns(4)
            h2 = c1.number_input("H2 (ppm)", min_value=0.0, value=0.0, step=0.1)
            ch4 = c2.number_input("CH4 (ppm)", min_value=0.0, value=0.0, step=0.1)
            c2h6 = c3.number_input("C2H6 (ppm)", min_value=0.0, value=0.0, step=0.1)
            c2h4 = c4.number_input("C2H4 (ppm)", min_value=0.0, value=0.0, step=0.1)

            c5, c6, c7, c8 = st.columns(4)
            c2h2 = c5.number_input("C2H2 (ppm)", min_value=0.0, value=0.0, step=0.1)
            co = c6.number_input("CO (ppm)", min_value=0.0, value=0.0, step=0.1)
            co2 = c7.number_input("CO2 (ppm)", min_value=0.0, value=0.0, step=0.1)
            ratio_o2_n2 = c8.number_input("O2/N2 Ratio (ppm/ppm)", min_value=0.0, max_value=1.0, value=0.32, step=0.01)

            st.markdown("**EXPANDED OIL ANALYSIS (IEC 60422 PARAMETERS):**")

            o1, o2_col, o3, o4 = st.columns(4)
            bdv = o1.number_input("BDV (kV)", min_value=0.0, value=50.0, step=0.1)
            acid = o2_col.number_input("Acid Number (mgKOH/g)", min_value=0.0, value=0.05, step=0.01)
            water = o3.number_input("Water Content (ppm)", min_value=0.0, value=12.0, step=0.1)
            ift = o4.number_input("IFT (mN/m)", min_value=0.0, value=30.0, step=0.1)

            o5, o6, o7, o8 = st.columns(4)
            ddf = o5.number_input("DDF / Tan Delta (90°C)", min_value=0.0, value=0.008, step=0.001)
            resistivity = o6.number_input("Resistivity Gohm.m (20°C)", min_value=0.0, value=70.0, step=1.0)
            colour_iso = o7.number_input("Oil Colour (ISO 2049 Scale)", min_value=0.5, max_value=8.0, value=1.5, step=0.5)
            sediment_sludge = o8.selectbox("Sediment / Sludge State", ["No", "Sediment", "Sludge"])

            o9, o10, o11, o12 = st.columns(4)
            corrosive_sulfur = o9.selectbox("Corrosive Sulphur", ["Non-Corrosive", "Corrosive"])
            particles_iso = o10.selectbox("Particles ISO 4406", ["Good", "Fair", "Poor"])
            inhibitor = o11.number_input("Inhibitor Content (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            passivator = o12.number_input("Passivator Content (mg/kg)", min_value=0.0, value=100.0, step=1.0)

            c_f1, c_f2 = st.columns(2)
            flash_point = c_f1.number_input("Flash Point (°C)", min_value=0.0, value=145.0, step=1.0)
            pcb_content = c_f2.number_input("PCB Content (mg/kg)", min_value=0.0, value=0.0, step=0.1)

            submit_btn = st.form_submit_button("EXECUTE EVALUATION & SAVE RECORD")

        if submit_btn:
            input_dict = {
                'ID_Trafo': trafo_id_input.strip(),
                'Tanggal_Uji': test_date,
                'H2': h2, 'CH4': ch4, 'C2H6': c2h6, 'C2H4': c2h4, 'C2H2': c2h2,
                'CO': co, 'CO2': co2, 'Ratio_O2_N2': ratio_o2_n2,
                'BDV': bdv, 'Acid': acid, 'Water': water, 'IFT': ift,
                'DDF': ddf, 'Resistivity': resistivity, 'Colour_ISO2049': colour_iso,
                'Sediment_Sludge': sediment_sludge, 'Corrosive_Sulphur': corrosive_sulfur,
                'Particles_ISO': particles_iso, 'Inhibitor_Content': inhibitor,
                'Passivator_Content': passivator, 'Flash_Point': flash_point,
                'PCB_Content': pcb_content, 'Status_Pemurnian': purification_status,
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
        The parameter values or gas growth rates entered exceed normal limits. Data cannot be committed directly to Master Database.

        **Before proceeding, verify the following field conditions:**
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
            st.success("Anomalous data verified and committed.")
            st.rerun()
            
        if col_c2.button("Cancel / Correct Input Data"):
            st.session_state['pending_data'] = None
            st.session_state['anomaly_reason'] = None
            st.info("Commit operation cancelled.")
            st.rerun()

# TAB 4: DATABASE MASTER & LEDGER
with tab_data:
    st.markdown("<div class='sap-title'>MASTER LEDGER & RECORD MANAGEMENT</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        with st.spinner("Processing analytical models and prognosis data..."):
            df_prognosis = calculate_prognosis_and_prediction(df_all)
            
        st.markdown("**Master Table Output (ReadOnly Grid)**")
        st.dataframe(df_prognosis, use_container_width=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            buffer_prog = io.BytesIO()
            with pd.ExcelWriter(buffer_prog, engine='openpyxl') as writer:
                df_prognosis.to_excel(writer, index=False, sheet_name='Data_Prognosis')
            
            st.download_button(
                label="EXPORT PROGNOSIS LEDGER (.XLSX)",
                data=buffer_prog.getvalue(),
                file_name=f"Transformer_Prognosis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col_exp2:
            buffer_meta = io.BytesIO()
            with pd.ExcelWriter(buffer_meta, engine='openpyxl') as writer:
                df_metadata.to_excel(writer, index=False, sheet_name='Nameplate_Metadata')
            
            st.download_button(
                label="EXPORT NAMEPLATE METADATA (.XLSX)",
                data=buffer_meta.getvalue(),
                file_name=f"Transformer_Nameplates_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.markdown("---")
        st.markdown("<div class='sap-title'>HISTORICAL RECORD DELETION CONTROL</div>", unsafe_allow_html=True)
        st.info("Notice: Single point of entry enforced. Direct cell editing is disabled. You can only remove historical observation rows.")
        
        delete_options = df_all.apply(
            lambda x: f"ID: {x['id']} | Transformer: {x['ID_Trafo']} | Date: {x['Tanggal_Uji']}", axis=1
        ).tolist()
        
        record_to_delete = st.selectbox("SELECT HISTORICAL RECORD TO PURGE", ["-- Select Record --"] + delete_options)
        
        if st.button("PURGE SELECTED HISTORICAL ROW"):
            if record_to_delete != "-- Select Record --":
                db_id = int(record_to_delete.split("|")[0].replace("ID:", "").strip())
                st.session_state['delete_confirm_id'] = db_id
            else:
                st.error("Please select a valid record.")
                
        if 'delete_confirm_id' in st.session_state and st.session_state['delete_confirm_id'] is not None:
            st.warning("SYSTEM VERIFICATION: Are you sure you want to permanently purge this row from SQLite Database?")
            col_del1, col_del2 = st.columns(2)
            if col_del1.button("CONFIRM PURGE RECORD"):
                delete_data(st.session_state['delete_confirm_id'])
                st.session_state['delete_confirm_id'] = None
                st.success("Record purged successfully.")
                st.rerun()
            if col_del2.button("CANCEL OPERATION"):
                st.session_state['delete_confirm_id'] = None
                st.rerun()
    else:
        st.info("No data available in database.")

# TAB 5: DIAGNOSTIC INSIGHTS
with tab_insights:
    st.markdown("<div class='sap-title'>DIAGNOSTIC INSIGHTS & TECHNICAL DATASHEET</div>", unsafe_allow_html=True)
    if not df_all.empty:
        df_prog_insight = calculate_prognosis_and_prediction(df_all)
        insight_trafo = st.selectbox("SELECT TRANSFORMER UNIT", df_prog_insight['ID_Trafo'].unique())
        
        df_insight_filtered = df_prog_insight[(df_prog_insight['ID_Trafo'] == insight_trafo) & (df_prog_insight['Tipe_Data'] == 'Historis')].sort_values('Tanggal_Uji').reset_index(drop=True)
        
        if not df_insight_filtered.empty:
            latest_record = df_insight_filtered.iloc[-1]
            
            meta_match = df_metadata[df_metadata['ID_Trafo'] == insight_trafo]
            if not meta_match.empty and pd.notna(meta_match.iloc[0]['Year_Manufactured']):
                y_manuf = meta_match.iloc[0]['Year_Manufactured']
                curr_test_year = pd.to_datetime(latest_record['Tanggal_Uji']).year
                disp_age = f"{curr_test_year - y_manuf} Years (Mfg: {y_manuf})"
                disp_manuf = meta_match.iloc[0]['Manufacturer']
                disp_cap = f"{meta_match.iloc[0]['Capacity_MVA']} MVA"
            else:
                disp_age = "Unknown"
                disp_manuf = "N/A"
                disp_cap = "N/A"

            fault_rows = df_insight_filtered[df_insight_filtered['Status_DGA'] != 'Normal']
            if not fault_rows.empty:
                first_fault_date = fault_rows.iloc[0]['Tanggal_Uji']
                first_fault_status = fault_rows.iloc[0]['Status_DGA']
                first_fault_text = f"{first_fault_date} (Initial Classification: {first_fault_status})"
            else:
                first_fault_text = "NONE DETECTED (Continuous Normal Status)"
            
            st.markdown(f"""
            <table class='hmi-table'>
                <tr>
                    <td class='hmi-label'>UNIT ID / MANUFACTURER</td>
                    <td class='hmi-value' colspan='3'>{latest_record['ID_Trafo']} | {disp_manuf} ({disp_cap})</td>
                </tr>
                <tr>
                    <td class='hmi-label'>OPERATIONAL AGE (DYNAMIC)</td>
                    <td class='hmi-value'>{disp_age}</td>
                    <td class='hmi-label'>LAST TEST DATE</td>
                    <td class='hmi-value'>{latest_record['Tanggal_Uji']}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>CURRENT DGA FAULT STATUS</td>
                    <td class='hmi-value' style='color: #D97706;'>{latest_record['Status_DGA']}</td>
                    <td class='hmi-label'>FIRST FAULT EVER DETECTED</td>
                    <td class='hmi-value'>{first_fault_text}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>SOLID PAPER INSULATION CONDITION</td>
                    <td class='hmi-value' colspan='3'>{latest_record['Paper_Status']}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>OIL MAINTENANCE ACTION (IEC 60422)</td>
                    <td class='hmi-value' colspan='3' style='color: #0D9488;'>{latest_record['OA_Recommendation']}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>ACTION TRIGGER / THRESHOLD EXCEEDED</td>
                    <td class='hmi-value' colspan='3' style='color: #F3F4F6;'>{latest_record['OA_Recommendation_Reason']}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>6-MONTH PROGNOSIS & ESCALATION TRAJECTORY</td>
                    <td class='hmi-value' colspan='3'>{latest_record['Prognosis_DGA']}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data.")
    else:
        st.info("No data available.")

# TAB 6: TREND ANALYSIS & GAUGES
with tab_trend:
    st.markdown("<div class='sap-title'>HMI GAS DYNAMICS & INSTRUMENT GAUGES</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        trafo_filter = st.selectbox("SELECT TRANSFORMER UNIT", df_all['ID_Trafo'].unique(), key="trend_selector")
        
        df_graph = calculate_prognosis_and_prediction(df_all)
        df_filtered = df_graph[df_graph['ID_Trafo'] == trafo_filter].sort_values('Tanggal_Uji').reset_index(drop=True)
        
        fig_gas = go.Figure()
        
        gas_list = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
        colors = ['#38BDF8', '#F59E0B', '#10B981', '#EF4444', '#A855F7']
        
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
                line_width=1.5,
                line_dash="dash",
                line_color="#D97706"
            )
            fig_gas.add_annotation(
                x=last_hist_date,
                y=1.02,
                yref="paper",
                text=f"HISTORICAL BASELINE ({last_hist_date})",
                showarrow=False,
                font=dict(color="#D97706", size=11, family="IBM Plex Mono")
            )

        fig_gas.update_layout(
            title=dict(
                text=f"DGA GAS CONCENTRATION EVOLUTION & FORECAST - {trafo_filter}",
                font=dict(family="IBM Plex Mono", color="#F3F4F6", size=14)
            ),
            xaxis_title="TEST DATE / FORECAST",
            yaxis_title="CONCENTRATION (PPM)",
            template="plotly_dark",
            paper_bgcolor="#1A2332",
            plot_bgcolor="#121926",
            font=dict(family="IBM Plex Mono", color="#9CA3AF"),
            height=420,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_gas, use_container_width=True)

        st.markdown("<div class='sap-title'>CURRENT DGA GAS CONCENTRATIONS (EXCLUDING O2/N2 RATIO)</div>", unsafe_allow_html=True)

        if not df_hist_only.empty:
            last_row = df_hist_only.iloc[-1]
            
            def fmt_val(v, unit="ppm"):
                return f"{float(v):.1f} {unit}" if pd.notna(v) else f"0.0 {unit}"

            st.markdown(f"""
            <table class='hmi-table'>
                <tr>
                    <td class='hmi-label'>HYDROGEN (H2)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('H2'))}</td>
                    <td class='hmi-label'>METHANE (CH4)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('CH4'))}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>ETHANE (C2H6)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('C2H6'))}</td>
                    <td class='hmi-label'>ETHYLENE (C2H4)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('C2H4'))}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>ACETYLENE (C2H2)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('C2H2'))}</td>
                    <td class='hmi-label'>CARBON MONOXIDE (CO)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('CO'))}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>CARBON DIOXIDE (CO2)</td>
                    <td class='hmi-value'>{fmt_val(last_row.get('CO2'))}</td>
                    <td class='hmi-label'>OBSERVATION DATE</td>
                    <td class='hmi-value'>{last_row.get('Tanggal_Uji')}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sap-title'>CURRENT EXTENDED OIL ANALYSIS (OA) CATEGORICAL STATUS LEDGER</div>", unsafe_allow_html=True)

        if not df_hist_only.empty:
            st.markdown(f"""
            <table class='hmi-table'>
                <tr>
                    <td class='hmi-label'>SEDIMENT / SLUDGE STATE</td>
                    <td class='hmi-value'>{last_row.get('Sediment_Sludge', 'No')}</td>
                    <td class='hmi-label'>CORROSIVE SULPHUR</td>
                    <td class='hmi-value'>{last_row.get('Corrosive_Sulphur', 'Non-Corrosive')}</td>
                </tr>
                <tr>
                    <td class='hmi-label'>PARTICLES ISO 4406</td>
                    <td class='hmi-value'>{last_row.get('Particles_ISO', 'Good')}</td>
                    <td class='hmi-label'>PURIFICATION STATUS</td>
                    <td class='hmi-value'>{last_row.get('Status_Pemurnian', 'Normal')}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sap-title'>PHYSICAL & CHEMICAL OIL GAUGES (LAST OBSERVATION - CATEGORY C <72.5 kV)</div>", unsafe_allow_html=True)
        
        if not df_hist_only.empty:
            last_oa = df_hist_only.iloc[-1]
            
            # GAUGE ROW 1: BDV, ACID, WATER, IFT
            g1, g2, g3, g4 = st.columns(4)
            
            bdv_val = float(last_oa.get('BDV', 0.0))
            fig_bdv = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bdv_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "BDV (kV) [Min 40]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 30], 'color': "#7F1D1D"},
                        {'range': [30, 40], 'color': "#78350F"},
                        {'range': [40, 100], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_bdv.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g1.plotly_chart(fig_bdv, use_container_width=True)

            acid_val = float(last_oa.get('Acid', 0.0))
            fig_acid = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acid_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Acid (mgKOH/g) [Max 0.15]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 0.5], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 0.15], 'color': "#064E3B"},
                        {'range': [0.15, 0.30], 'color': "#78350F"},
                        {'range': [0.30, 0.5], 'color': "#7F1D1D"}
                    ]
                }
            ))
            fig_acid.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g2.plotly_chart(fig_acid, use_container_width=True)

            water_val = float(last_oa.get('Water', 0.0))
            fig_water = go.Figure(go.Indicator(
                mode="gauge+number",
                value=water_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Water (ppm) [Max 30]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 60], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 30], 'color': "#064E3B"},
                        {'range': [30, 40], 'color': "#78350F"},
                        {'range': [40, 60], 'color': "#7F1D1D"}
                    ]
                }
            ))
            fig_water.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g3.plotly_chart(fig_water, use_container_width=True)

            ift_val = float(last_oa.get('IFT', 0.0))
            fig_ift = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ift_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "IFT (mN/m) [Min 28]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 50], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 22], 'color': "#7F1D1D"},
                        {'range': [22, 28], 'color': "#78350F"},
                        {'range': [28, 50], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_ift.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g4.plotly_chart(fig_ift, use_container_width=True)

            # GAUGE ROW 2: DDF, RESISTIVITY, COLOUR ISO 2049, INHIBITOR
            g5, g6, g7, g8 = st.columns(4)

            ddf_val = float(last_oa.get('DDF', 0.0))
            fig_ddf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ddf_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "DDF/Tan Delta [Max 0.10]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 0.60], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 0.10], 'color': "#064E3B"},
                        {'range': [0.10, 0.50], 'color': "#78350F"},
                        {'range': [0.50, 0.60], 'color': "#7F1D1D"}
                    ]
                }
            ))
            fig_ddf.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g5.plotly_chart(fig_ddf, use_container_width=True)

            res_val = float(last_oa.get('Resistivity', 0.0))
            fig_res = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Resistivity Gohm.m [Min 60]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 4], 'color': "#7F1D1D"},
                        {'range': [4, 60], 'color': "#78350F"},
                        {'range': [60, 100], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_res.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g6.plotly_chart(fig_res, use_container_width=True)

            col_val = float(last_oa.get('Colour_ISO2049', 1.0))
            fig_col = go.Figure(go.Indicator(
                mode="gauge+number",
                value=col_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Colour (ISO 2049) [Max 2.0]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0.5, 8.0], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0.5, 2.0], 'color': "#064E3B"},
                        {'range': [2.0, 3.5], 'color': "#78350F"},
                        {'range': [3.5, 8.0], 'color': "#7F1D1D"}
                    ]
                }
            ))
            fig_col.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g7.plotly_chart(fig_col, use_container_width=True)

            inh_val = float(last_oa.get('Inhibitor_Content', 0.0))
            fig_inh = go.Figure(go.Indicator(
                mode="gauge+number",
                value=inh_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Inhibitor Content (%) [Min 60]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 40], 'color': "#7F1D1D"},
                        {'range': [40, 60], 'color': "#78350F"},
                        {'range': [60, 100], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_inh.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g8.plotly_chart(fig_inh, use_container_width=True)

            # GAUGE ROW 3: PASSIVATOR, FLASH POINT, PCB
            g9, g10, g11, _ = st.columns(4)

            pass_val = float(last_oa.get('Passivator_Content', 0.0))
            fig_pass = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pass_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Passivator (mg/kg) [Min 70]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 150], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 50], 'color': "#7F1D1D"},
                        {'range': [50, 70], 'color': "#78350F"},
                        {'range': [70, 150], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_pass.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g9.plotly_chart(fig_pass, use_container_width=True)

            flash_val = float(last_oa.get('Flash_Point', 0.0))
            fig_flash = go.Figure(go.Indicator(
                mode="gauge+number",
                value=flash_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "Flash Point (°C) [Min 135]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 180], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 120], 'color': "#7F1D1D"},
                        {'range': [120, 135], 'color': "#78350F"},
                        {'range': [135, 180], 'color': "#064E3B"}
                    ]
                }
            ))
            fig_flash.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g10.plotly_chart(fig_flash, use_container_width=True)

            pcb_val = float(last_oa.get('PCB_Content', 0.0))
            fig_pcb = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pcb_val,
                domain={'y': [0.0, 0.75]},
                title={'text': "PCB Content (mg/kg) [Max 2.0]", 'font': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 10}},
                number={'font': {'family': 'IBM Plex Mono', 'color': '#F3F4F6', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 10], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#9CA3AF', 'size': 8}},
                    'bar': {'color': "#D97706"},
                    'bgcolor': "#121926",
                    'bordercolor': "#2A364F",
                    'steps': [
                        {'range': [0, 2.0], 'color': "#064E3B"},
                        {'range': [2.0, 5.0], 'color': "#78350F"},
                        {'range': [5.0, 10.0], 'color': "#7F1D1D"}
                    ]
                }
            ))
            fig_pcb.update_layout(height=230, paper_bgcolor="#1A2332", margin=dict(l=20, r=20, t=60, b=10))
            g11.plotly_chart(fig_pcb, use_container_width=True)
    else:
        st.info("No data available.")
