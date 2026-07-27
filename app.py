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

# ==========================================
# 1. KONFIGURASI HALAMAN (SAP CLASSIC STYLE)
# ==========================================
st.set_page_config(
    page_title="Transformer DGA Diagnostic System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Tampilan Industrial / SAP Retro
st.markdown("""
    <style>
    .main { background-color: #F1F5F9; }
    .stButton>button {
        background-color: #1E293B;
        color: white;
        border-radius: 2px;
        border: 1px solid #0F172A;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #334155;
        color: white;
    }
    .sap-title {
        background-color: #1E293B;
        padding: 10px 16px;
        color: white;
        font-family: monospace;
        font-size: 16px;
        font-weight: bold;
        border-bottom: 3px solid #0EA5E9;
        margin-bottom: 15px;
    }
    .sap-header {
        background-color: #0F172A;
        padding: 14px 20px;
        color: white;
        font-family: sans-serif;
        font-size: 18px;
        font-weight: bold;
        border-bottom: 4px solid #0EA5E9;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE ENGINE (SQLite + Auto Seed CSV)
# ==========================================
DB_FILE = "dga_database.db"
CSV_SEED_FILE = "Data_Uji_Trafo.csv"
JALUR_MODEL_LOKAL = r'model_dga_7classes_v2.pkl'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Skema SQLite Murni Tanpa Furan dan DP_Kertas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Trafo TEXT NOT NULL,
            Tanggal_Uji TEXT NOT NULL,
            H2 REAL, CH4 REAL, C2H6 REAL, C2H4 REAL, C2H2 REAL,
            CO REAL, CO2 REAL, O2 REAL, N2 REAL,
            BDV REAL, Acid REAL, Water REAL, IFT REAL,
            Status_Pemurnian TEXT,
            Is_Anomali TEXT DEFAULT 'Tidak'
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM tabel_master")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(CSV_SEED_FILE):
        df_csv = pd.read_csv(CSV_SEED_FILE)
        df_csv['Tanggal_Uji'] = pd.to_datetime(df_csv['Tanggal_Uji']).dt.strftime('%Y-%m-%d')
        df_csv['Status_Pemurnian'] = df_csv['Status_Pemurnian'].fillna('Normal')
        df_csv['Is_Anomali'] = 'Tidak'
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

init_db()

# ==========================================
# 3. GATEKEEPER & VALIDASI ANOMALI INPUT
# ==========================================
def cek_anomali(df_historis, id_trafo, tgl_input, data_input):
    df_trafo = df_historis[df_historis['ID_Trafo'] == id_trafo].sort_values('Tanggal_Uji_DT')
    is_pemurnian_valid = data_input['Status_Pemurnian'] in ["Reconditioning", "Reclaiming", "Oil Replacement"]
    is_trafo_baru = df_trafo.empty
    
    terlampaui = []
    if data_input['H2'] > 5000: terlampaui.append(f"H2: {data_input['H2']} ppm (Batas: 5000 ppm)")
    if data_input['CH4'] > 3000: terlampaui.append(f"CH4: {data_input['CH4']} ppm (Batas: 3000 ppm)")
    if data_input['C2H4'] > 2000: terlampaui.append(f"C2H4: {data_input['C2H4']} ppm (Batas: 2000 ppm)")
    if data_input['C2H6'] > 1000: terlampaui.append(f"C2H6: {data_input['C2H6']} ppm (Batas: 1000 ppm)")
    if data_input['C2H2'] > 500: terlampaui.append(f"C2H2: {data_input['C2H2']} ppm (Batas: 500 ppm)")
    if data_input['CO'] > 5000: terlampaui.append(f"CO: {data_input['CO']} ppm (Batas: 5000 ppm)")
    if data_input['CO2'] > 20000: terlampaui.append(f"CO2: {data_input['CO2']} ppm (Batas: 20000 ppm)")
    if data_input['BDV'] > 100: terlampaui.append(f"BDV: {data_input['BDV']} kV (Batas: 100 kV)")
    if data_input['Water'] > 100: terlampaui.append(f"Water: {data_input['Water']} ppm (Batas: 100 ppm)")
    if data_input['Acid'] > 1.0: terlampaui.append(f"Acid: {data_input['Acid']} mgKOH/g (Batas: 1.0 mgKOH/g)")
    
    if terlampaui:
        return True, "Nilai parameter melebihi ambang batas fisik uji laboratorium:\n- " + "\n- ".join(terlampaui)
    
    if data_input['C2H2'] > 0 and (data_input['H2'] == 0 or data_input['CH4'] == 0):
        return True, "Inkonsistensi Fisika DGA: C2H2 terdeteksi tanpa adanya pembentukan gas dasar H2 atau CH4."
    
    if is_pemurnian_valid:
        if (data_input['H2'] > 50 or data_input['CH4'] > 50 or data_input['C2H4'] > 30 or data_input['C2H2'] > 2):
            return True, f"Konsentrasi gas terlalu tinggi untuk kondisi trafo pasca pemurnian ({data_input['Status_Pemurnian']})."
        return False, "Normal"
        
    if is_trafo_baru:
        if (data_input['H2'] > 150 or data_input['CH4'] > 120 or data_input['C2H6'] > 65 or data_input['C2H4'] > 50 or data_input['C2H2'] > 2):
            return True, "Nilai gas awal trafo baru melebihi batas kondisi baseline IEEE."
        return False, "Normal"

    rec_last = df_trafo.iloc[-1]
    tgl_last = rec_last['Tanggal_Uji_DT']
    selisih_hari = (tgl_input - tgl_last).days
    
    last_h2 = rec_last['H2'] if pd.notnull(rec_last['H2']) else 0
    last_ch4 = rec_last['CH4'] if pd.notnull(rec_last['CH4']) else 0
    
    if last_h2 == 0 and data_input['H2'] >= 100:
        return True, f"Lonjakan ekstrem H2 dari baseline 0 ppm menjadi {data_input['H2']} ppm."
    if last_ch4 == 0 and data_input['CH4'] >= 80:
        return True, f"Lonjakan ekstrem CH4 dari baseline 0 ppm menjadi {data_input['CH4']} ppm."
        
    if selisih_hari > 0:
        rate_h2 = ((data_input['H2'] - last_h2) / selisih_hari) * 30.43
        rate_ch4 = ((data_input['CH4'] - last_ch4) / selisih_hari) * 30.43
        rate_c2h4 = ((data_input['C2H4'] - (rec_last['C2H4'] if pd.notnull(rec_last['C2H4']) else 0)) / selisih_hari) * 30.43
        rate_c2h2 = ((data_input['C2H2'] - (rec_last['C2H2'] if pd.notnull(rec_last['C2H2']) else 0)) / selisih_hari) * 30.43
        
        detail_rate = []
        if rate_h2 > 15: detail_rate.append(f"Laju H2: {rate_h2:.1f} ppm/bulan (Batas: 15 ppm/bulan)")
        if rate_ch4 > 12: detail_rate.append(f"Laju CH4: {rate_ch4:.1f} ppm/bulan (Batas: 12 ppm/bulan)")
        if rate_c2h4 > 5: detail_rate.append(f"Laju C2H4: {rate_c2h4:.1f} ppm/bulan (Batas: 5 ppm/bulan)")
        if rate_c2h2 > 0.2: detail_rate.append(f"Laju C2H2: {rate_c2h2:.1f} ppm/bulan (Batas: 0.2 ppm/bulan)")
        
        if detail_rate:
            return True, "Laju pertumbuhan gas bulanan (Gas Growth Rate) melebihi ambang batas wajar:\n- " + "\n- ".join(detail_rate)

    return False, "Normal"

# ==========================================
# 4. ENGINE EKSPLISIT POWER BI (100% FAITHFUL)
# ==========================================
def dapatkan_ambang_ieee_30tahun(o2_n2_ratio):
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

def dapatkan_minimum_duval(ch4, c2h4, c2h2):
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

def hitung_prognosis_dan_prediksi(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
        
    df = df_raw.copy()
    
    # Normalisasi desimal koma & numerik murni
    kolom_numerik = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'O2', 'N2', 'BDV', 'Acid', 'Water', 'IFT']
    for col in kolom_numerik:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Status_Pemurnian' in df.columns:
        df['Status_Pemurnian'] = df['Status_Pemurnian'].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    else:
        df['Status_Pemurnian'] = np.nan

    df['Tanggal_Uji_DT'] = pd.to_datetime(df['Tanggal_Uji'], errors='coerce')

    kolom_forecast = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
    semua_gas_7 = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
    langkah_prediksi = 6
    hasil_keseluruhan = []

    # ARIMA Forecasting Engine
    for trafo in df['ID_Trafo'].unique():
        df_trafo = df[df['ID_Trafo'] == trafo].copy()
        df_trafo['Tipe_Data'] = 'Historis'
        df_trafo = df_trafo.sort_values('Tanggal_Uji_DT').reset_index(drop=True)

        for gas in kolom_forecast:
            df_trafo[f'GGR_{gas}'] = df_trafo[gas].diff().fillna(0)
            df_trafo[f'GGR_{gas}'] = df_trafo[f'GGR_{gas}'].apply(lambda x: round(max(x, 0), 2))

        waktu_terakhir_DT = df_trafo['Tanggal_Uji_DT'].iloc[-1]
        data_terakhir = df_trafo.iloc[-1].copy()

        tgl_dua_tahun_lalu = waktu_terakhir_DT - relativedelta(years=2)
        df_temporal = df_trafo[df_trafo['Tanggal_Uji_DT'] >= tgl_dua_tahun_lalu].copy()

        idx_pemurnian = df_temporal[df_temporal['Status_Pemurnian'].isin(['Oil Replacement', 'Reclaiming', 'Reconditioning'])].index
        
        freeze_mode = False
        if not idx_pemurnian.empty:
            pos_terakhir_pemurnian = idx_pemurnian[-1]
            tgl_pemurnian = df_temporal.loc[pos_terakhir_pemurnian, 'Tanggal_Uji_DT']
            df_era_baru = df_temporal[df_temporal['Tanggal_Uji_DT'] >= tgl_pemurnian].copy()
            
            umur_era_baru = (waktu_terakhir_DT.year - tgl_pemurnian.year) * 12 + (waktu_terakhir_DT.month - tgl_pemurnian.month)
            if len(df_era_baru) < 6 or umur_era_baru < 4:
                freeze_mode = True
                data_train_arima = df_era_baru.copy()
            else:
                data_train_arima = df_era_baru.tail(6).copy()
        else:
            data_train_arima = df_temporal.tail(6).copy()

        prediksi_masa_depan = {col: [] for col in kolom_forecast}
        tanggal_prediksi = [(waktu_terakhir_DT + relativedelta(months=i+1)).strftime('%Y-%m-%d') for i in range(langkah_prediksi)]

        for col in kolom_forecast:
            deret_waktu = data_train_arima[col].values
            if freeze_mode or len(deret_waktu) < 3:
                ramalan = [deret_waktu[-1] for _ in range(langkah_prediksi)]
            else:
                try:
                    model = ARIMA(deret_waktu, order=(1, 1, 0))
                    model_fit = model.fit()
                    ramalan = model_fit.forecast(steps=langkah_prediksi)
                except:
                    selisih_rata2 = np.mean(np.diff(deret_waktu[-3:])) if len(deret_waktu) > 1 else 0
                    ramalan = [deret_waktu[-1] + (selisih_rata2 * (i+1)) for i in range(langkah_prediksi)]
            
            ramalan = np.maximum(ramalan, deret_waktu[-1])
            prediksi_masa_depan[col] = ramalan

        df_trafo_prediksi_list = []
        for i in range(langkah_prediksi):
            baris_baru = data_terakhir.copy()
            baris_baru['Tanggal_Uji'] = tanggal_prediksi[i]
            baris_baru['Tanggal_Uji_DT'] = pd.to_datetime(tanggal_prediksi[i])
            baris_baru['Tipe_Data'] = 'Prediksi'
            baris_baru['Status_Pemurnian'] = np.nan
            
            for col in kolom_forecast:
                baris_baru[col] = round(prediksi_masa_depan[col][i], 2)
                
            nilai_bulan_lalu = df_trafo_prediksi_list[i-1] if i > 0 else data_terakhir
            for gas in kolom_forecast:
                ggr = baris_baru[gas] - (nilai_bulan_lalu[gas] if pd.notna(nilai_bulan_lalu[gas]) else 0)
                baris_baru[f'GGR_{gas}'] = round(max(ggr, 0), 2)
                
            df_trafo_prediksi_list.append(baris_baru)

        df_trafo_prediksi = pd.DataFrame(df_trafo_prediksi_list)
        hasil_keseluruhan.append(df_trafo)
        hasil_keseluruhan.append(df_trafo_prediksi)

    df_master = pd.concat(hasil_keseluruhan, ignore_index=True)

    # RANDOM FOREST AI MODEL EXECUTION
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
    status_kertas_list = []
    rekomendasi_oa_list = []

    df_master['Tanggal_Uji_DT'] = pd.to_datetime(df_master['Tanggal_Uji'], errors='coerce')
    df_master = df_master.sort_values(['ID_Trafo', 'Tanggal_Uji_DT']).reset_index(drop=True)

    trafo_freeze_status = {}

    # HYBRID LOGIC GATEKEEPER & OA RECOMMENDATION
    for idx, row in df_master.iterrows():
        trafo_id = row['ID_Trafo']

        status_p_str = str(row['Status_Pemurnian']).strip() if pd.notna(row['Status_Pemurnian']) else ""
        if status_p_str in ['Oil Replacement', 'Reclaiming', 'Reconditioning']:
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

        t1_limits, t2_limits, t3_delta_limits, t4_rate_limits = dapatkan_ambang_ieee_30tahun(ratio_o2_n2)

        exceed_t1_any = any((pd.notna(row[g]) and row[g] > t1_limits[g]) for g in semua_gas_7 if g in row)
        exceed_t2_any = any((pd.notna(row[g]) and row[g] > t2_limits[g]) for g in semua_gas_7 if g in row)

        is_rates_anomali = False
        
        if not is_currently_frozen:
            if idx > 0 and df_master.loc[idx, 'ID_Trafo'] == df_master.loc[idx-1, 'ID_Trafo']:
                selisih_hari = (row['Tanggal_Uji_DT'] - df_master.loc[idx-1, 'Tanggal_Uji_DT']).days
                if selisih_hari > 0:
                    selisih_hari_efektif = max(selisih_hari, 3)
                    for g in semua_gas_7:
                        if g in row and pd.notna(row[g]) and pd.notna(df_master.loc[idx-1, g]):
                            selisih_ppm = row[g] - df_master.loc[idx-1, g]
                            laju_tahunan = (selisih_ppm / selisih_hari_efektif) * 365.25

                            if g == 'C2H2' and selisih_ppm >= 0.5: is_rates_anomali = True
                            elif g != 'C2H2' and selisih_ppm > t3_delta_limits[g]: is_rates_anomali = True

                            if laju_tahunan > t4_rate_limits[g]: is_rates_anomali = True

        vonis_ai = row['Vonis_AI_Mentah']

        if not exceed_t1_any and not is_rates_anomali and not exceed_t2_any:
            status_ieee = "Status 1"
            status_dga_final = "Normal"
        elif exceed_t2_any or is_rates_anomali:
            status_ieee = "Status 3"
            ch4_v = row['CH4'] if pd.notna(row['CH4']) else 0
            c2h4_v = row['C2H4'] if pd.notna(row['C2H4']) else 0
            c2h2_v = row['C2H2'] if pd.notna(row['C2H2']) else 0
            
            vonis_fisika_pasti = dapatkan_minimum_duval(ch4_v, c2h4_v, c2h2_v)

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

        # Evaluasi Kertas Isolasi Padat Berbasis CO & CO2 Murni (IEEE C57.104 Annex D.8)
        co = row['CO'] if 'CO' in row and pd.notna(row['CO']) else 0
        co2 = row['CO2'] if 'CO2' in row and pd.notna(row['CO2']) else 0
        ratio_co2_co = co2 / (co + 1e-5)

        if co > 1000 and ratio_co2_co < 3:
            if exceed_t1_any or is_rates_anomali:
                status_kertas_list.append("Indication of Fault Involving Solid Paper Insulation")
            else:
                status_kertas_list.append("Oil Oxidation (Restricted O2) - Paper Insulation Intact")
        elif co2 > 10000 and ratio_co2_co > 20:
            status_kertas_list.append("Slow Degradation of Paper Insulation")
        else:
            status_kertas_list.append("Normal")

        # Rekomendasi Oil Analysis (IEC 60422)
        bdv = row['BDV'] if pd.notna(row['BDV']) else 0
        acid = row['Acid'] if pd.notna(row['Acid']) else 0
        water = row['Water'] if pd.notna(row['Water']) else 0
        ift = row['IFT'] if pd.notna(row['IFT']) else 0

        if is_currently_frozen:
            rekomendasi_oa = "New oil baseline detected. System under intensive post-maintenance monitoring (Freeze Mode Active)."
        elif acid > 0.30 or ift < 16:
            rekomendasi_oa = "OA RECOMMENDATION: MANDATORY TOTAL OIL REPLACEMENT (IEC 60422). Pressure Flushing Required."
        elif acid >= 0.15 or (20 <= ift <= 24):
            rekomendasi_oa = "OA RECOMMENDATION: Oil Reclaiming Required (Fuller's Earth)."
        elif bdv < 50 or water > 25:
            rekomendasi_oa = "OA RECOMMENDATION: Oil Reconditioning Required (Filtration & Vacuum Dehydration)."
        else:
            rekomendasi_oa = "Insulating oil physical conditions normal (IEC 60422)."

        rekomendasi_oa_list.append(rekomendasi_oa)

    df_master['DGA_Status_IEEE'] = dga_status_ieee_list
    df_master['Status_DGA'] = status_dga_final_list
    df_master['Status_Kertas'] = status_kertas_list
    df_master['Rekomendasi_OA'] = rekomendasi_oa_list

    # Temporal Prognosis Engine
    df_master['Prognosis_DGA'] = ""
    df_master['Severity_Level'] = df_master['Status_DGA'].apply(get_severity_score)

    for trafo_id in df_master['ID_Trafo'].unique():
        idx_trafo = df_master[df_master['ID_Trafo'] == trafo_id].index
        list_idx = list(idx_trafo)

        for pos, idx in enumerate(list_idx):
            status_sekarang = df_master.loc[idx, 'Status_DGA']
            severity_sekarang = df_master.loc[idx, 'Severity_Level']
            tanggal_baris = df_master.loc[idx, 'Tanggal_Uji_DT']

            daftar_eskalasi = []
            severity_terlewati = severity_sekarang
            status_tercatat = set([status_sekarang])

            for idx_depan in list_idx[pos + 1:]:
                severity_depan = df_master.loc[idx_depan, 'Severity_Level']
                tgl_depan = df_master.loc[idx_depan, 'Tanggal_Uji_DT']
                status_depan = df_master.loc[idx_depan, 'Status_DGA']

                if pd.notna(tgl_depan) and pd.notna(tanggal_baris):
                    jarak_hari = (tgl_depan - tanggal_baris).days
                    selisih_bulan = int(round(jarak_hari / 30.43))
                    if selisih_bulan == 0 and jarak_hari > 15: selisih_bulan = 1

                    if severity_depan > severity_terlewati and status_depan not in status_tercatat:
                        daftar_eskalasi.append((status_depan, selisih_bulan))
                        severity_terlewati = severity_depan
                        status_tercatat.add(status_depan)

            if severity_sekarang == 0 or status_sekarang == "Normal":
                if daftar_eskalasi:
                    bagian = [f"Potential progression to {st} within {bl} month(s)" for st, bl in daftar_eskalasi]
                    kesimpulan = "Normal. " + " | ".join(bagian)
                else:
                    kesimpulan = "Normal (Conditions predicted to remain stable)"
            else:
                if daftar_eskalasi:
                    bagian = [f"potential escalation to {st} within {bl} month(s)" for st, bl in daftar_eskalasi]
                    kesimpulan = f"{status_sekarang} detected | " + " | ".join(bagian)
                else:
                    kesimpulan = f"{status_sekarang} detected | Fault condition persisting"

            df_master.loc[idx, 'Prognosis_DGA'] = kesimpulan

    return df_master.drop(columns=['Tanggal_Uji_DT', 'Severity_Level', 'Vonis_AI_Mentah'], errors='ignore')

# ==========================================
# 5. ANTARMUKA UTAMA (LAYOUT TABBED)
# ==========================================
st.markdown("<div class='sap-header'>TRANSFORMER DIGITAL TWIN & DGA DIAGNOSTIC SYSTEM</div>", unsafe_allow_html=True)

df_all = load_data()

tab1, tab2, tab3 = st.tabs(["[1] Form Input Data", "[2] Master Data & Prognosis", "[3] Visualisasi Tren"])

# ------------------------------------------
# TAB 1: FORM INPUT & GATEKEEPER ANOMALI
# ------------------------------------------
with tab1:
    st.markdown("<div class='sap-title'>INPUT PARAMETER UJI LAB DGA & MINYAK</div>", unsafe_allow_html=True)
    
    list_trafo_existing = sorted(df_all['ID_Trafo'].unique().tolist()) if not df_all.empty else []
    options_trafo = list_trafo_existing + ["+ Tambah Trafo Baru..."]
    
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        selected_trafo_option = st.selectbox("ID Trafo", options_trafo)
        if selected_trafo_option == "+ Tambah Trafo Baru...":
            id_trafo_input = st.text_input("Nama ID Trafo Baru", placeholder="Contoh: Main_Transformer_06")
        else:
            id_trafo_input = selected_trafo_option

    with col_meta2:
        tgl_uji = st.date_input("Tanggal Uji Lab", datetime.now())

    with col_meta3:
        status_pemurnian = st.selectbox("Status Pemurnian Minyak", ["Normal", "Reconditioning", "Reclaiming", "Oil Replacement"])

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

        st.markdown("**Karakteristik Fisik Minyak (Oil Analysis):**")
        o1, o2_col, o3, o4 = st.columns(4)
        bdv = o1.number_input("BDV (kV)", min_value=0.0, value=0.0, step=0.1)
        acid = o2_col.number_input("Acid Number (mgKOH/g)", min_value=0.0, value=0.0, step=0.01)
        water = o3.number_input("Water Content (ppm)", min_value=0.0, value=0.0, step=0.1)
        ift = o4.number_input("IFT (mN/m)", min_value=0.0, value=0.0, step=0.1)

        btn_submit = st.form_submit_button("EVALUASI & SIMPAN DATA")

    if btn_submit:
        if not id_trafo_input or id_trafo_input.strip() == "":
            st.error("ID Trafo tidak boleh kosong. Mohon pilih atau masukan ID Trafo.")
        else:
            data_input = {
                'ID_Trafo': id_trafo_input.strip(),
                'Tanggal_Uji': tgl_uji,
                'H2': h2, 'CH4': ch4, 'C2H6': c2h6, 'C2H4': c2h4, 'C2H2': c2h2,
                'CO': co, 'CO2': co2, 'O2': o2, 'N2': n2,
                'Acid': acid, 'BDV': bdv, 'IFT': ift, 'Water': water,
                'Status_Pemurnian': status_pemurnian,
                'Is_Anomali': 'Tidak'
            }
            
            is_anomali, detail_alasan = cek_anomali(df_all, id_trafo_input.strip(), pd.to_datetime(tgl_uji), data_input)
            
            if is_anomali:
                st.session_state['pending_data'] = data_input
                st.session_state['detail_alasan'] = detail_alasan
            else:
                insert_data(data_input)
                st.success(f"Data Uji untuk ID Trafo '{id_trafo_input.strip()}' berhasil disimpan ke database.")
                st.rerun()

    # Panel Verifikasi Anomali
    if 'pending_data' in st.session_state and st.session_state['pending_data'] is not None:
        st.markdown("---")
        st.error(f"**PERINGATAN DETEKSI ANOMALI INPUT:**\n\n{st.session_state['detail_alasan']}")
        
        st.info("""
        Nilai parameter DGA atau laju pertumbuhan gas yang Anda masukkan terdeteksi melebihi ambang batas wajar (anomali). Data ini tidak dapat disimpan ke Tabel Master secara langsung.

        **Sebelum menyimpan data, mohon verifikasi faktor lapangan berikut:**
        1. Apakah prosedur pengambilan sampel minyak sudah sesuai standar?
        2. Apakah ada potensi kontaminasi sampel (misalnya: botol sampel kotor atau tidak tertutup rapat)?
        3. Apakah terdapat faktor pengotor eksternal saat proses pengujian di laboratorium?
        """)
        
        col_confirm1, col_confirm2 = st.columns(2)
        if col_confirm1.button("Tetap Simpan Data (Dipastikan Valid Hasil Uji)"):
            p_data = st.session_state['pending_data']
            p_data['Is_Anomali'] = 'Ya'
            insert_data(p_data)
            st.session_state['pending_data'] = None
            st.session_state['detail_alasan'] = None
            st.success("Data anomali berhasil terverifikasi dan disimpan.")
            st.rerun()
            
        if col_confirm2.button("Batal / Perbaiki Input Data"):
            st.session_state['pending_data'] = None
            st.session_state['detail_alasan'] = None
            st.info("Penyimpanan dibatalkan. Silakan periksa kembali parameter input.")
            st.rerun()

# ------------------------------------------
# TAB 2: MASTER DATA, PROGNOSIS & EXPORT EXCEL
# ------------------------------------------
with tab2:
    st.markdown("<div class='sap-title'>MASTER DATA & HASIL DIAGNOSA PROGNOSIS (ARIMA + AI)</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        with st.spinner("Menghitung model ARIMA, Rekomendasi OA, Status Kertas & Prognosis DGA..."):
            df_prognosis = hitung_prognosis_dan_prediksi(df_all)
            
        st.dataframe(df_prognosis, use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_prognosis.to_excel(writer, index=False, sheet_name='Prognosis_DGA')
        
        st.download_button(
            label="DOWNLOAD TABEL PROGNOSIS COMPLETE (.XLSX)",
            data=buffer.getvalue(),
            file_name=f"Hasil_Prognosis_DGA_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Belum ada data tersimpan di database.")

# ------------------------------------------
# TAB 3: VISUALISASI TREN & GAUGE OA
# ------------------------------------------
with tab3:
    st.markdown("<div class='sap-title'>GRAFIK TREN KONSENTRASI GAS & GAUGE KARAKTERISTIK MINYAK (OA)</div>", unsafe_allow_html=True)
    
    if not df_all.empty:
        trafo_filter = st.selectbox("Pilih ID Trafo", df_all['ID_Trafo'].unique())
        
        df_graph = hitung_prognosis_dan_prediksi(df_all)
        df_filtered = df_graph[df_graph['ID_Trafo'] == trafo_filter].sort_values('Tanggal_Uji').reset_index(drop=True)
        
        # GRAFIK KONTINU DISAMBUNGKAN DENGAN MARKER HISTORIS TERAKHIR
        fig_gas = go.Figure()
        
        gas_list = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        df_hist_only = df_filtered[df_filtered['Tipe_Data'] == 'Historis']
        tgl_terakhir_historis = df_hist_only['Tanggal_Uji'].iloc[-1] if not df_hist_only.empty else None

        for idx_g, gas in enumerate(gas_list):
            if gas in df_filtered.columns:
                fig_gas.add_trace(go.Scatter(
                    x=df_filtered['Tanggal_Uji'],
                    y=df_filtered[gas],
                    mode='lines+markers',
                    name=gas,
                    line=dict(color=colors[idx_g % len(colors)]),
                    marker=dict(size=6)
                ))

        if tgl_terakhir_historis:
            fig_gas.add_vline(
                x=tgl_terakhir_historis,
                line_width=2,
                line_dash="dash",
                line_color="#EF4444"
            )
            fig_gas.add_annotation(
                x=tgl_terakhir_historis,
                y=1.02,
                yref="paper",
                text=f"Batas Historis ({tgl_terakhir_historis})",
                showarrow=False,
                font=dict(color="#EF4444", size=12, family="monospace")
            )

        fig_gas.update_layout(
            title=f"Evolusi & Forecast Konsentrasi Gas DGA (H2, CH4, C2H6, C2H4, C2H2) - {trafo_filter}",
            xaxis_title="Tanggal Uji / Prediksi",
            yaxis_title="Konsentrasi (ppm)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig_gas, use_container_width=True)

        # INDIKATOR GAUGE PARAMETER OIL ANALYSIS (DATA HISTORIS TERAKHIR)
        st.markdown("<div class='sap-title'>INDIKATOR GAUGE OA HISTORIS TERAKHIR</div>", unsafe_allow_html=True)
        
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
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FCA5A5"},
                        {'range': [30, 50], 'color': "#FEF08A"},
                        {'range': [50, 100], 'color': "#BBF7D0"}
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
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 0.15], 'color': "#BBF7D0"},
                        {'range': [0.15, 0.30], 'color': "#FEF08A"},
                        {'range': [0.30, 0.5], 'color': "#FCA5A5"}
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
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 25], 'color': "#BBF7D0"},
                        {'range': [25, 40], 'color': "#FEF08A"},
                        {'range': [40, 60], 'color': "#FCA5A5"}
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
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 16], 'color': "#FCA5A5"},
                        {'range': [16, 24], 'color': "#FEF08A"},
                        {'range': [24, 50], 'color': "#BBF7D0"}
                    ]
                }
            ))
            fig_ift.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            g4.plotly_chart(fig_ift, use_container_width=True)

            st.info(f"**STATUS REKOMENDASI OA (IEC 60422):**\n{last_oa['Rekomendasi_OA']}")
    else:
        st.info("Silakan masukan data pada Tab 1 untuk menampilkan grafik.")