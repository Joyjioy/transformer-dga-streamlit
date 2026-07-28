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

import ui_components as ui

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Transformer Substation Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui.load_css("style.css")

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
            Manufacturer TEXT, Serial_Number TEXT, Year_Manufactured INTEGER,
            Model_Type TEXT, Capacity_MVA REAL, Phase_Count INTEGER,
            Nominal_Voltage_kV TEXT, Nominal_Current_A TEXT, Frequency_Hz REAL,
            Vector_Group TEXT, Impedance_Pct REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Trafo TEXT NOT NULL, Tanggal_Uji TEXT NOT NULL,
            H2 REAL, CH4 REAL, C2H6 REAL, C2H4 REAL, C2H2 REAL, CO REAL, CO2 REAL, Ratio_O2_N2 REAL,
            BDV REAL, Acid REAL, Water REAL, IFT REAL, DDF REAL, Resistivity REAL, Colour_ISO2049 REAL,
            Sediment_Sludge TEXT, Corrosive_Sulphur TEXT, Particles_ISO TEXT,
            Inhibitor_Content REAL, Passivator_Content REAL, Flash_Point REAL, PCB_Content REAL,
            Status_Pemurnian TEXT, Is_Anomali TEXT DEFAULT 'No'
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
            INSERT OR IGNORE INTO tabel_trafo_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, meta_records)
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM tabel_master")
    if cursor.fetchone()[0] == 0 and os.path.exists(CSV_SEED_FILE):
        df_csv = pd.read_csv(CSV_SEED_FILE)
        df_csv['Tanggal_Uji'] = pd.to_datetime(df_csv['Tanggal_Uji']).dt.strftime('%Y-%m-%d')
        df_csv['Status_Pemurnian'] = df_csv['Status_Pemurnian'].fillna('Normal')
        df_csv['Is_Anomali'] = 'No'
        df_csv = df_csv.replace({np.nan: None})

        for _, row in df_csv.iterrows():
            data_dict = {
                'ID_Trafo': str(row['ID_Trafo']), 'Tanggal_Uji': str(row['Tanggal_Uji']),
                'H2': float(row['H2']) if pd.notna(row.get('H2')) else None,
                'CH4': float(row['CH4']) if pd.notna(row.get('CH4')) else None,
                'C2H6': float(row['C2H6']) if pd.notna(row.get('C2H6')) else None,
                'C2H4': float(row['C2H4']) if pd.notna(row.get('C2H4')) else None,
                'C2H2': float(row['C2H2']) if pd.notna(row.get('C2H2')) else None,
                'CO': float(row['CO']) if pd.notna(row.get('CO')) else None,
                'CO2': float(row['CO2']) if pd.notna(row.get('CO2')) else None,
                'Ratio_O2_N2': float(row['Ratio_O2_N2']) if pd.notna(row.get('Ratio_O2_N2')) else None,
                'BDV': float(row['BDV']) if pd.notna(row.get('BDV')) else None,
                'Acid': float(row['Acid']) if pd.notna(row.get('Acid')) else None,
                'Water': float(row['Water']) if pd.notna(row.get('Water')) else None,
                'IFT': float(row['IFT']) if pd.notna(row.get('IFT')) else None,
                'DDF': float(row['DDF']) if pd.notna(row.get('DDF')) else None,
                'Resistivity': float(row['Resistivity']) if pd.notna(row.get('Resistivity')) else None,
                'Colour_ISO2049': float(row['Colour_ISO2049']) if pd.notna(row.get('Colour_ISO2049')) else None,
                'Sediment_Sludge': str(row.get('Sediment_Sludge')) if pd.notna(row.get('Sediment_Sludge')) else None,
                'Corrosive_Sulphur': str(row.get('Corrosive_Sulphur')) if pd.notna(row.get('Corrosive_Sulphur')) else None,
                'Particles_ISO': str(row.get('Particles_ISO')) if pd.notna(row.get('Particles_ISO')) else None,
                'Inhibitor_Content': float(row['Inhibitor_Content']) if pd.notna(row.get('Inhibitor_Content')) else None,
                'Passivator_Content': float(row['Passivator_Content']) if pd.notna(row.get('Passivator_Content')) else None,
                'Flash_Point': float(row['Flash_Point']) if pd.notna(row.get('Flash_Point')) else None,
                'PCB_Content': float(row['PCB_Content']) if pd.notna(row.get('PCB_Content')) else None,
                'Status_Pemurnian': str(row['Status_Pemurnian']),
                'Is_Anomali': str(row['Is_Anomali'])
            }
            insert_data(data_dict)

    conn.close()

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM tabel_master ORDER BY Tanggal_Uji ASC", conn)
    conn.close()
    df['Tanggal_Uji_DT'] = pd.to_datetime(df.get('Tanggal_Uji', pd.Series(dtype='str')), errors='coerce')
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
        INSERT OR REPLACE INTO tabel_trafo_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(meta_dict.values()))
    conn.commit()
    conn.close()

def insert_data(data_dict):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    kolom_urutan = [
        'ID_Trafo', 'Tanggal_Uji', 'H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'Ratio_O2_N2',
        'BDV', 'Acid', 'Water', 'IFT', 'DDF', 'Resistivity', 'Colour_ISO2049',
        'Sediment_Sludge', 'Corrosive_Sulphur', 'Particles_ISO',
        'Inhibitor_Content', 'Passivator_Content', 'Flash_Point', 'PCB_Content',
        'Status_Pemurnian', 'Is_Anomali'
    ]

    nilai_terurut = tuple(data_dict.get(k) for k in kolom_urutan)

    cursor.execute(f"""
        INSERT INTO tabel_master ({', '.join(kolom_urutan)})
        VALUES ({', '.join(['?'] * len(kolom_urutan))})
    """, nilai_terurut)
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
    if df_history.empty:
        df_trafo = pd.DataFrame()
    else:
        df_trafo = df_history[df_history['ID_Trafo'] == trafo_id].sort_values('Tanggal_Uji_DT')

    is_purification_valid = input_data.get('Status_Pemurnian') in ["Reconditioning", "Reclaiming", "Oil Replacement"]
    is_new_trafo = df_trafo.empty

    def get_num(key):
        val = input_data.get(key)
        return float(val) if val is not None and pd.notna(val) else 0.0

    val_h2 = get_num('H2')
    val_ch4 = get_num('CH4')
    val_c2h6 = get_num('C2H6')
    val_c2h4 = get_num('C2H4')
    val_c2h2 = get_num('C2H2')
    val_bdv = get_num('BDV')
    val_water = get_num('Water')
    val_acid = get_num('Acid')
    val_colour = get_num('Colour_ISO2049')

    exceeded = []
    if val_h2 > 5000: exceeded.append(f"H2: {val_h2} ppm")
    if val_ch4 > 3000: exceeded.append(f"CH4: {val_ch4} ppm")
    if val_c2h4 > 2000: exceeded.append(f"C2H4: {val_c2h4} ppm")
    if val_c2h6 > 1000: exceeded.append(f"C2H6: {val_c2h6} ppm")
    if val_c2h2 > 500: exceeded.append(f"C2H2: {val_c2h2} ppm")
    if val_bdv > 100: exceeded.append(f"BDV: {val_bdv} kV")
    if val_water > 100: exceeded.append(f"Water: {val_water} ppm")
    if val_acid > 1.0: exceeded.append(f"Acid: {val_acid} mgKOH/g")
    if val_colour > 8.0: exceeded.append(f"Colour: {val_colour} ISO 2049")

    if exceeded:
        return True, "Parameter values exceed physical laboratory thresholds:\n- " + "\n- ".join(exceeded)

    if val_c2h2 > 0 and (val_h2 == 0 or val_ch4 == 0):
        return True, "DGA Physical Inconsistency: C2H2 detected without base gas H2 or CH4 generation."

    if is_purification_valid:
        if (val_h2 > 50 or val_ch4 > 50 or val_c2h4 > 30 or val_c2h2 > 2):
            return True, f"Gas concentrations are abnormally high for a post-purification condition ({input_data.get('Status_Pemurnian')})."
        return False, "Normal"

    if is_new_trafo:
        if (val_h2 > 150 or val_ch4 > 120 or val_c2h6 > 65 or val_c2h4 > 50 or val_c2h2 > 2):
            return True, "Initial gas values for new transformer exceed IEEE baseline limits."
        return False, "Normal"

    if not df_trafo.empty:
        rec_last = df_trafo.iloc[-1]
        last_date = rec_last['Tanggal_Uji_DT']
        days_diff = (input_date - last_date).days

        last_h2 = rec_last['H2'] if pd.notnull(rec_last['H2']) else 0.0
        last_ch4 = rec_last['CH4'] if pd.notnull(rec_last['CH4']) else 0.0

        if days_diff > 0:
            rate_h2 = ((val_h2 - last_h2) / days_diff) * 30.43
            rate_ch4 = ((val_ch4 - last_ch4) / days_diff) * 30.43

            rate_details = []
            if rate_h2 > 15: rate_details.append(f"H2 Rate: {rate_h2:.1f} ppm/month")
            if rate_ch4 > 12: rate_details.append(f"CH4 Rate: {rate_ch4:.1f} ppm/month")
            if rate_details:
                return True, "Monthly Gas Growth Rate exceeds acceptable thresholds:\n- " + "\n- ".join(rate_details)

    return False, "Normal"

def get_ieee_thresholds(o2_n2_ratio, age_years, period_months=12):
    is_sealed = (o2_n2_ratio is not None and o2_n2_ratio <= 0.2)

    if age_years is None or age_years <= 0: age_cat = 'unknown'
    elif 1 <= age_years <= 9: age_cat = '1_9'
    elif 10 <= age_years <= 30: age_cat = '10_30'
    else: age_cat = 'gt_30'

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
        t4_rate = {'H2': 25 if period_months < 10 else 10, 'CH4': 10 if period_months < 10 else 4, 'C2H6': 15 if period_months < 10 else 9, 'C2H4': 10 if period_months < 10 else 5, 'C2H2': 0.01, 'CO': 100, 'CO2': 1750 if period_months < 10 else 1000}
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
        t4_rate = {'H2': 50 if period_months < 10 else 20, 'CH4': 15 if period_months < 10 else 3, 'C2H6': 2 if period_months < 10 else 3, 'C2H4': 7, 'C2H2': 0.01, 'CO': 200 if period_months < 10 else 80, 'CO2': 1000 if period_months < 10 else 800}

    return t1_map[age_cat], t2_map[age_cat], t3_delta, t4_rate

def get_duval_minimum(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: 
        return "Normal"
    p_ch4 = (ch4 / total) * 100
    p_c2h4 = (c2h4 / total) * 100
    p_c2h2 = (c2h2 / total) * 100

    if p_ch4 >= 98:
        return "PD"
    
    if p_c2h2 < 4:
        if p_c2h4 < 20:
            return "T1"
        elif 20 <= p_c2h4 < 50:
            return "T2"
        else:
            return "T3"
            
    if p_c2h4 < 23:
        if p_c2h2 >= 13:
            return "D1"
        else:
            return "T1"
    else:
        if p_c2h2 >= 29:
            return "D2"
        elif 4 <= p_c2h2 < 29:
            if p_c2h4 >= 50 and p_c2h2 < 15:
                return "T3"
            elif 40 <= p_c2h4 < 71 and p_c2h2 >= 15:
                return "D2"
            else:
                return "DT"
    return "DT"

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

def create_hmi_gauge(val, title, min_val, max_val, steps_config):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': title, 'font': {'family': 'IBM Plex Mono', 'color': '#0F172A', 'size': 13}},
        number={'font': {'family': 'IBM Plex Mono', 'color': '#0F172A', 'size': 28}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickfont': {'family': 'IBM Plex Mono', 'color': '#475569', 'size': 10}},
            'bar': {'color': "#0F172A", 'thickness': 0.25},
            'bgcolor': "#FFFFFF",
            'bordercolor': "#CBD5E1",
            'steps': steps_config
        }
    ))
    fig.update_layout(height=230, paper_bgcolor="#FFFFFF", margin=dict(l=25, r=25, t=50, b=20))
    return fig

def plot_duval_triangle1(ch4, c2h4, c2h2, trafo_id):
    total = ch4 + c2h4 + c2h2
    if total == 0:
        p_ch4, p_c2h4, p_c2h2 = 0.0, 0.0, 0.0
    else:
        p_ch4 = (ch4 / total) * 100
        p_c2h4 = (c2h4 / total) * 100
        p_c2h2 = (c2h2 / total) * 100

    fig = go.Figure()

    zones = [
        {'name': 'PD', 'a': [100, 98, 98, 100], 'b': [0, 2, 0, 0], 'c': [0, 0, 2, 0],
         'color': 'rgba(147, 197, 253, 0.4)', 'text_pos': (99.0, 0.5, 0.5)},
        {'name': 'T1', 'a': [98, 98, 80, 76, 96, 98], 'b': [2, 0, 0, 4, 4, 2], 'c': [0, 2, 20, 20, 0, 0],
         'color': 'rgba(254, 240, 138, 0.4)', 'text_pos': (88.0, 1.5, 10.5)},
        {'name': 'T2', 'a': [80, 50, 46, 76, 80], 'b': [0, 0, 4, 4, 0], 'c': [20, 50, 50, 20, 20],
         'color': 'rgba(253, 186, 116, 0.4)', 'text_pos': (63.0, 2.0, 35.0)},
        {'name': 'T3', 'a': [50, 0, 0, 35, 50], 'b': [0, 0, 15, 15, 0], 'c': [50, 100, 85, 50, 50],
         'color': 'rgba(248, 113, 113, 0.4)', 'text_pos': (20.0, 5.0, 75.0)},
        {'name': 'D1', 'a': [0, 96, 73, 0, 0], 'b': [100, 4, 4, 77, 100], 'c': [0, 0, 23, 23, 0],
         'color': 'rgba(192, 132, 252, 0.4)', 'text_pos': (30.0, 60.0, 10.0)},
        {'name': 'D2', 'a': [0, 73, 56, 31, 0, 0], 'b': [77, 4, 4, 29, 29, 77], 'c': [23, 23, 40, 40, 71, 23],
         'color': 'rgba(244, 63, 94, 0.5)', 'text_pos': (20.0, 45.0, 35.0)},
        {'name': 'DT', 'a': [73, 46, 35, 0, 0, 31, 56, 73], 'b': [4, 4, 15, 15, 29, 29, 4, 4],
         'c': [23, 50, 50, 85, 71, 40, 40, 23], 'color': 'rgba(203, 213, 225, 0.5)', 'text_pos': (40.0, 15.0, 45.0)}
    ]

    for z in zones:
        fig.add_trace(go.Scatterternary({
            'mode': 'lines', 'a': z['a'], 'b': z['b'], 'c': z['c'],
            'fill': 'toself', 'fillcolor': z['color'], 'line': {'color': '#334155', 'width': 1},
            'name': z['name'], 'hoverinfo': 'name', 'showlegend': True
        }))
        tp = z['text_pos']
        fig.add_trace(go.Scatterternary({
            'mode': 'text', 'a': [tp[0]], 'b': [tp[1]], 'c': [tp[2]],
            'text': [f"<b>{z['name']}</b>"],
            'textfont': {'size': 12, 'color': '#0F172A', 'family': 'IBM Plex Mono'},
            'showlegend': False, 'hoverinfo': 'none'
        }))

    fig.add_trace(go.Scatterternary({
        'mode': 'markers+text', 'a': [p_ch4], 'b': [p_c2h2], 'c': [p_c2h4],
        'text': [f"<b>{trafo_id}</b>"], 'textposition': "top center",
        'textfont': {'size': 12, 'color': '#000000', 'family': 'IBM Plex Mono'},
        'marker': {'symbol': 'diamond', 'color': '#FFFF00', 'size': 16, 'line': {'width': 2.5, 'color': '#000000'}},
        'name': 'Current Observation', 'hoverinfo': 'text'
    }))

    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': {'text': '% CH4', 'font': {'color': '#0F172A', 'size': 13}}, 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': {'text': '% C2H2', 'font': {'color': '#0F172A', 'size': 13}}, 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': {'text': '% C2H4', 'font': {'color': '#0F172A', 'size': 13}}, 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'bgcolor': '#FFFFFF'
        },
        'paper_bgcolor': '#FFFFFF',
        'title': dict(text=f"DUVAL TRIANGLE 1 FAULT GEOMETRY MAP — {trafo_id}", font=dict(family="IBM Plex Mono", color="#0F172A", size=14)),
        'height': 600, 'margin': dict(l=40, r=40, t=50, b=40),
        'legend': dict(orientation="h", y=-0.1, x=0.05)
    })

    return fig, p_ch4, p_c2h4, p_c2h2

def calculate_prognosis_and_prediction(df_raw):
    if df_raw.empty: return pd.DataFrame()
    df = df_raw.copy()
    df_meta = load_metadata()

    if 'Status_Pemurnian' not in df.columns:
        df['Status_Pemurnian'] = 'Normal'

    df['Tanggal_Uji_DT'] = pd.to_datetime(df['Tanggal_Uji'], errors='coerce')
    df = df.sort_values(['ID_Trafo', 'Tanggal_Uji_DT']).reset_index(drop=True)

    numeric_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'Ratio_O2_N2', 'BDV', 'Acid', 'Water', 'IFT', 'DDF', 'Resistivity', 'Colour_ISO2049', 'Inhibitor_Content', 'Passivator_Content', 'Flash_Point', 'PCB_Content']
    categorical_columns = ['Sediment_Sludge', 'Corrosive_Sulphur', 'Particles_ISO']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    gas_cols = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']

    is_purify_row = df['Status_Pemurnian'].fillna('').astype(str).str.strip().isin(['Oil Replacement', 'Reclaiming', 'Reconditioning'])

    for col in gas_cols:
        if col in df.columns:
            df.loc[is_purify_row & df[col].isna(), col] = 0.0

    df['Is_Purification_Event'] = is_purify_row.astype(int)
    df['Era_Pemurnian'] = df.groupby('ID_Trafo')['Is_Purification_Event'].cumsum()

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df.groupby(['ID_Trafo', 'Era_Pemurnian'])[col].ffill()

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df.groupby(['ID_Trafo', 'Era_Pemurnian'])[col].ffill()

    df = df.drop(columns=['Is_Purification_Event', 'Era_Pemurnian'], errors='ignore')

    forecast_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
    all_7_gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
    prediction_steps = 6
    final_results = []

    for trafo in df['ID_Trafo'].unique():
        df_trafo = df[df['ID_Trafo'] == trafo].copy()
        df_trafo['Tipe_Data'] = 'Historical'
        df_trafo = df_trafo.sort_values('Tanggal_Uji_DT').reset_index(drop=True)

        last_time_dt = df_trafo['Tanggal_Uji_DT'].iloc[-1]
        last_data = df_trafo.iloc[-1].copy()

        two_years_ago = last_time_dt - relativedelta(years=2)
        df_temporal = df_trafo[df_trafo['Tanggal_Uji_DT'] >= two_years_ago].copy()

        purification_idx = df_temporal[df_temporal['Status_Pemurnian'].isin(['Oil Replacement', 'Reclaiming', 'Reconditioning'])].index

        freeze_mode = False
        if not purification_idx.empty:
            last_p_date = df_temporal.loc[purification_idx[-1], 'Tanggal_Uji_DT']
            df_new_era = df_temporal[df_temporal['Tanggal_Uji_DT'] >= last_p_date].copy()
            if len(df_new_era) < 6:
                freeze_mode = True
                data_train_arima = df_new_era.copy()
            else:
                data_train_arima = df_new_era.tail(6).copy()
        else:
            data_train_arima = df_temporal.tail(6).copy()

        future_predictions = {col: [] for col in forecast_columns}
        prediction_dates = [(last_time_dt + relativedelta(months=i+1)).strftime('%Y-%m-%d') for i in range(prediction_steps)]

        for col in forecast_columns:
            ts_series = data_train_arima[col].ffill().bfill()
            time_series = ts_series.values if not ts_series.empty else np.array([0.0])

            if freeze_mode or len(time_series) < 3:
                forecast = [time_series[-1] if len(time_series) > 0 else 0.0 for _ in range(prediction_steps)]
            else:
                try:
                    model = ARIMA(time_series, order=(1, 1, 0))
                    forecast = model.fit().forecast(steps=prediction_steps)
                except:
                    avg_diff = np.mean(np.diff(time_series[-3:])) if len(time_series) > 1 else 0
                    forecast = [time_series[-1] + (avg_diff * (i+1)) for i in range(prediction_steps)]

            future_predictions[col] = np.maximum(forecast, time_series[-1] if len(time_series) > 0 else 0.0)

        df_trafo_prediction_list = []
        for i in range(prediction_steps):
            new_row = last_data.copy()
            new_row['Tanggal_Uji'] = prediction_dates[i]
            new_row['Tanggal_Uji_DT'] = pd.to_datetime(prediction_dates[i])
            new_row['Tipe_Data'] = 'Forecast'
            new_row['Status_Pemurnian'] = np.nan
            for col in forecast_columns:
                new_row[col] = round(future_predictions[col][i], 2)
            df_trafo_prediction_list.append(new_row)

        final_results.extend([df_trafo, pd.DataFrame(df_trafo_prediction_list)])

    df_master = pd.concat(final_results, ignore_index=True)

    if os.path.exists(JALUR_MODEL_LOKAL):
        model_dga = joblib.load(JALUR_MODEL_LOKAL)
        try:
            gas_features = df_master[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']].ffill().bfill().fillna(0.0)
            df_master['Vonis_AI_Mentah'] = model_dga.predict(gas_features)
        except Exception:
            df_master['Vonis_AI_Mentah'] = 'Normal'
    else:
        df_master['Vonis_AI_Mentah'] = 'Normal'

    dga_status_ieee_list, status_dga_final_list, status_paper_list = [], [], []
    recommendation_oa_list, recommendation_oa_reason_list = [], []

    df_master['Tanggal_Uji_DT'] = pd.to_datetime(df_master['Tanggal_Uji'], errors='coerce')
    df_master = df_master.sort_values(['ID_Trafo', 'Tanggal_Uji_DT']).reset_index(drop=True)

    trafo_freeze_status = {}

    for idx, row in df_master.iterrows():
        trafo_id = row['ID_Trafo']
        meta_match = df_meta[df_meta['ID_Trafo'] == trafo_id]
        calculated_age = (row['Tanggal_Uji_DT'].year - meta_match.iloc[0]['Year_Manufactured']) if not meta_match.empty else None

        o2_n2_ratio = row.get('Ratio_O2_N2', 0.32)
        if pd.isna(o2_n2_ratio): o2_n2_ratio = 0.32

        t1_limits, t2_limits, t3_delta_limits, t4_rate_limits = get_ieee_thresholds(o2_n2_ratio, calculated_age)

        exceed_t1_any = any((pd.notna(row[g]) and row[g] > t1_limits[g]) for g in all_7_gases if g in row)
        exceed_t2_any = any((pd.notna(row[g]) and row[g] > t2_limits[g]) for g in all_7_gases if g in row)

        is_rates_anomali = False
        if idx > 0 and df_master.loc[idx, 'ID_Trafo'] == df_master.loc[idx-1, 'ID_Trafo']:
            days_diff = (row['Tanggal_Uji_DT'] - df_master.loc[idx-1, 'Tanggal_Uji_DT']).days
            if days_diff > 0:
                for g in all_7_gases:
                    if pd.notna(row[g]) and pd.notna(df_master.loc[idx-1, g]):
                        diff_ppm = row[g] - df_master.loc[idx-1, g]
                        annual_rate = (diff_ppm / max(days_diff, 3)) * 365.25
                        if (g == 'C2H2' and diff_ppm >= 0.5) or (g != 'C2H2' and diff_ppm > t3_delta_limits[g]) or (annual_rate > t4_rate_limits[g]):
                            is_rates_anomali = True

        if not exceed_t1_any and not is_rates_anomali and not exceed_t2_any:
            status_ieee, status_dga_final = "Status 1", "Normal"
        elif exceed_t2_any or is_rates_anomali:
            status_ieee = "Status 3"
            c_ch4 = row['CH4'] if pd.notna(row['CH4']) else 0.0
            c_c2h4 = row['C2H4'] if pd.notna(row['C2H4']) else 0.0
            c_c2h2 = row['C2H2'] if pd.notna(row['C2H2']) else 0.0
            status_dga_final = get_duval_minimum(c_ch4, c_c2h4, c_c2h2)
        else:
            status_ieee, status_dga_final = "Status 2", "Caution"

        dga_status_ieee_list.append(status_ieee)
        status_dga_final_list.append(status_dga_final)

        co = row['CO'] if pd.notna(row['CO']) else 0.0
        co2 = row['CO2'] if pd.notna(row['CO2']) else 0.0
        ratio_co2_co = co2 / (co + 1e-5)
        if co > 1000 and ratio_co2_co < 3:
            status_paper_list.append("Indication of Fault Involving Solid Paper Insulation" if (exceed_t1_any or is_rates_anomali) else "Oil Oxidation (Restricted O2)")
        elif co2 > 10000 and ratio_co2_co > 20:
            status_paper_list.append("Slow Degradation of Paper Insulation")
        else:
            status_paper_list.append("Normal")

        status_p_str = str(row.get('Status_Pemurnian', '')).strip()
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
                (df_master['Tipe_Data'] == 'Historical')
            ]
            if len(sub_df) < 6 or umb < 4:
                is_currently_frozen = True

        bdv = float(row['BDV']) if pd.notna(row['BDV']) else None
        acid = float(row['Acid']) if pd.notna(row['Acid']) else None
        water = float(row['Water']) if pd.notna(row['Water']) else None
        ift = float(row['IFT']) if pd.notna(row['IFT']) else None
        ddf = float(row['DDF']) if pd.notna(row['DDF']) else None
        resistivity = float(row['Resistivity']) if pd.notna(row['Resistivity']) else None
        colour = float(row['Colour_ISO2049']) if pd.notna(row['Colour_ISO2049']) else None
        sediment_sludge = str(row.get('Sediment_Sludge')).strip() if pd.notna(row.get('Sediment_Sludge')) else ""
        corrosive_sulfur = str(row.get('Corrosive_Sulphur')).strip() if pd.notna(row.get('Corrosive_Sulphur')) else ""

        if is_currently_frozen:
            rec_oa = "OA STATUS: FREEZE MODE ACTIVE (MONITORING NEW ERA)"
            reason_str = "New oil baseline detected. System under intensive post-maintenance monitoring."
        elif all(v is None for v in [bdv, acid, water, ift]):
            rec_oa = "OA STATUS: DATA NOT TESTED (N/A)"
            reason_str = "Insufficient oil physical parameters recorded for Category C evaluation."
        else:
            bdv_v = bdv if bdv is not None else 0.0
            acid_v = acid if acid is not None else 0.0
            water_v = water if water is not None else 0.0
            ift_v = ift if ift is not None else 0.0
            ddf_v = ddf if ddf is not None else 0.0
            res_v = resistivity if resistivity is not None else 0.0
            col_v = colour if colour is not None else 1.0

            reasons = []
            if acid_v > 0.30: reasons.append(f"Acid Number critical ({acid_v:.2f} mgKOH/g > 0.30 Limit)")
            if 0 < ift_v < 22: reasons.append(f"Interfacial Tension critical ({ift_v:.1f} mN/m < 22 Limit)")
            if col_v >= 4.0: reasons.append(f"Colour Scale critical ({col_v:.1f} ISO 2049 >= 4.0 Limit)")
            if acid_v > 0.20 and ddf_v > 0.50 and 0 < res_v < 4: reasons.append("Severe Combined Aging Detected")

            if reasons:
                rec_oa = "OA RECOMMENDATION: MANDATORY TOTAL OIL REPLACEMENT (IEC 60422)"
                reason_str = " | ".join(reasons) + " Full Oil Replacement Required."
            elif corrosive_sulfur == "Corrosive":
                rec_oa = "OA RECOMMENDATION: PASSIVATION OR RECLAIMING REQUIRED (IEC 60422)"
                reason_str = "Corrosive Sulphur detected. Risk of copper sulfide deposition."
            else:
                reclaim_reasons = []
                if acid_v >= 0.15: reclaim_reasons.append(f"Acid elevated ({acid_v:.2f} mgKOH/g >= 0.15 Limit)")
                if 22 <= ift_v <= 28: reclaim_reasons.append(f"IFT degraded ({ift_v:.1f} mN/m)")
                if ddf_v > 0.10: reclaim_reasons.append(f"DDF/Tan Delta high ({ddf_v:.3f} > 0.10 Limit)")
                if 0 < res_v < 60: reclaim_reasons.append(f"Resistivity low ({res_v:.1f} Gohm.m < 60 Limit)")
                if col_v > 2.0: reclaim_reasons.append(f"Oil Colour degraded ({col_v:.1f} ISO 2049 > 2.0 Limit)")
                if sediment_sludge == "Sludge": reclaim_reasons.append("Precipitable Sludge detected")

                if reclaim_reasons:
                    rec_oa = "OA RECOMMENDATION: OIL RECLAIMING REQUIRED (FULLER'S EARTH)"
                    reason_str = " | ".join(reclaim_reasons) + " Fuller's Earth Adsorption needed."
                else:
                    recond_reasons = []
                    if 0 < bdv_v < 40: recond_reasons.append(f"Breakdown Voltage low ({bdv_v:.1f} kV < 40 Limit)")
                    if water_v > 30: recond_reasons.append(f"Water Content high ({water_v:.1f} ppm > 30 Limit)")

                    if recond_reasons:
                        rec_oa = "OA RECOMMENDATION: OIL RECONDITIONING REQUIRED (FILTRATION)"
                        reason_str = " | ".join(recond_reasons) + " Vacuum Dehydration & Filtration needed."
                    else:
                        rec_oa = "OA STATUS: NORMAL OPERATIONAL CONDITION (IEC 60422)"
                        reason_str = "All active oil parameters within normal limits (Category C)."

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
                    month_diff = max(1, int(round((future_date - row_date).days / 30.43)))
                    if future_severity > max_severity_passed and future_status not in recorded_status:
                        escalation_list.append((future_status, month_diff))
                        max_severity_passed = future_severity
                        recorded_status.add(future_status)

            if current_severity == 0 or current_status == "Normal":
                conclusion = "Normal. " + " | ".join([f"Potential progression to {st} within {mo} mo" for st, mo in escalation_list]) if escalation_list else "Normal (Stable)"
            else:
                conclusion = f"{current_status} detected | " + " | ".join([f"Potential escalation to {st} within {mo} mo" for st, mo in escalation_list]) if escalation_list else f"{current_status} detected"

            df_master.loc[idx, 'Prognosis_DGA'] = conclusion

    return df_master.drop(columns=['Tanggal_Uji_DT', 'Severity_Level', 'Vonis_AI_Mentah'], errors='ignore')

ui.render_header("TRANSFORMER SUBSTATION MONITORING SYSTEM")

df_all = load_data()
df_metadata = load_metadata()

tab_home, tab_nameplate, tab_input, tab_data, tab_insights, tab_trend, tab_duval = st.tabs([
    "System Overview",
    "Transformer Identity & Nameplate",
    "Data Input Form",
    "Database Master & Ledger",
    "Diagnostic Insights",
    "HMI Trend Analysis",
    "Duval Triangle 1"
])

with tab_home:
    ui.render_title("SYSTEM ARCHITECTURE & TECHNICAL SPECIFICATIONS")
    st.markdown("""
    This platform operates as an enterprise **HMI Substation Monitoring System** engineered for high-voltage power transformers across all age categories.

    The diagnostic engine combines physics-based domain standards with temporal machine learning:
    *   **Dynamic IEEE C57.104-2019 Standard:** Evaluates gas limits based on dynamically calculated operating age and O2/N2 ratio.
    *   **Duval Triangle 1 Geometry:** Identifies exact fault types (T1, T2, T3 Thermal Faults or PD, D1, D2 Electrical Discharges).
    *   **Solid Paper Insulation Evaluation:** Assesses cellulose paper degradation through Carbon Monoxide and Carbon Dioxide gas ratios.
    *   **Expanded IEC 60422 Oil Analysis (OA) Category C (< 72.5 kV):** Formulates oil health status based on active decision parameters (BDV, Water, Acid, IFT, DDF, Resistivity, ISO 2049 Colour, Corrosive Sulphur, and Sludge).
    *   **ARIMA Temporal Forecasting:** Generates 6-month predictive gas trajectories.
    """)

with tab_nameplate:
    ui.render_title("TRANSFORMER NAMEPLATE REGISTRATION PANEL")
    st.dataframe(df_metadata, use_container_width=True)
    st.markdown("---")

    with st.form("nameplate_form"):
        col_n1, col_n2, col_n3 = st.columns(3)
        reg_id = col_n1.text_input("Transformer ID", placeholder="Example: Main_Transformer_06").strip()
        reg_manuf = col_n2.text_input("Manufacturer Name", placeholder="Example: ABB Power")
        reg_sn = col_n3.text_input("Serial Number", placeholder="Example: SN-2024-99")

        col_n4, col_n5, col_n6 = st.columns(3)
        reg_year = col_n4.number_input("Year Manufactured", min_value=1950, max_value=2026, value=None, placeholder="1950-2026", step=1)
        reg_model = col_n5.text_input("Model / Type Designation", placeholder="Example: ONAN-TR6")
        reg_mva = col_n6.number_input("Power Capacity (MVA)", min_value=0.1, value=None, placeholder="e.g. 30.0", step=0.5)

        col_n7, col_n8, col_n9 = st.columns(3)
        reg_phase = col_n7.selectbox("Number of Phases", [3, 1])
        reg_voltage = col_n8.text_input("Nominal Voltage (kV)", placeholder="Example: 150/20")
        reg_current = col_n9.text_input("Nominal Current (A)", placeholder="Example: 115/866")

        col_n10, col_n11, col_n12 = st.columns(3)
        reg_freq = col_n10.number_input("Frequency (Hz)", min_value=40.0, max_value=70.0, value=None, placeholder="e.g. 50.0", step=1.0)
        reg_vector = col_n11.text_input("Vector Group", placeholder="Example: YNd11")
        reg_impedance = col_n12.number_input("Short Circuit Impedance (%)", min_value=0.1, value=None, placeholder="e.g. 12.5", step=0.1)

        if st.form_submit_button("REGISTER NAMEPLATE SPECIFICATIONS"):
            if not reg_id:
                st.error("Transformer ID cannot be empty.")
            else:
                meta_dict = {
                    'ID_Trafo': reg_id, 'Manufacturer': reg_manuf, 'Serial_Number': reg_sn,
                    'Year_Manufactured': int(reg_year) if reg_year else 2010,
                    'Model_Type': reg_model,
                    'Capacity_MVA': float(reg_mva) if reg_mva else 0.0,
                    'Phase_Count': int(reg_phase), 'Nominal_Voltage_kV': reg_voltage, 'Nominal_Current_A': reg_current,
                    'Frequency_Hz': float(reg_freq) if reg_freq else 50.0,
                    'Vector_Group': reg_vector,
                    'Impedance_Pct': float(reg_impedance) if reg_impedance else 0.0
                }
                insert_metadata(meta_dict)
                st.success(f"Nameplate for '{reg_id}' registered successfully.")
                st.rerun()

with tab_input:
    ui.render_title("LABORATORY OBSERVATION INPUT PANEL")
    registered_trafos = sorted(df_metadata['ID_Trafo'].unique().tolist()) if not df_metadata.empty else []

    if not registered_trafos:
        st.warning("No transformers registered. Please register unit nameplate first.")
    else:
        if st.session_state.get('pending_data') is not None:
            st.error(f"**INPUT ANOMALY DETECTED:**\n\n{st.session_state.get('anomaly_reason')}")
            st.warning("Input values exceed physical/operational thresholds. Please select an action:")

            col_c1, col_c2 = st.columns(2)
            if col_c1.button("Force Save Data", type="primary"):
                p_data = st.session_state['pending_data']
                p_data['Is_Anomali'] = 'Yes'
                insert_data(p_data)
                st.session_state['pending_data'] = None
                st.session_state['anomaly_reason'] = None
                st.success("Anomalous data forcibly committed to database.")
                st.rerun()

            if col_c2.button("Cancel Operation"):
                st.session_state['pending_data'] = None
                st.session_state['anomaly_reason'] = None
                st.info("Input operation cancelled.")
                st.rerun()

        else:
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            trafo_id_input = col_meta1.selectbox("SELECT REGISTERED TRANSFORMER", registered_trafos)
            test_date = col_meta2.date_input("TEST DATE", datetime.now())
            purification_status = col_meta3.selectbox("OIL PURIFICATION STATUS", ["Normal", "Reconditioning", "Reclaiming", "Oil Replacement"])

            with st.form("dga_input_form"):
                st.markdown("### DISSOLVED GAS ANALYSIS (PPM) & O2/N2 RATIO")

                c1, c2, c3, c4 = st.columns(4)
                h2 = c1.number_input("H2 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                ch4 = c2.number_input("CH4 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                c2h6 = c3.number_input("C2H6 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                c2h4 = c4.number_input("C2H4 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)

                c5, c6, c7, c8 = st.columns(4)
                c2h2 = c5.number_input("C2H2 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                co = c6.number_input("CO (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                co2 = c7.number_input("CO2 (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                ratio_o2_n2 = c8.number_input("O2/N2 Ratio", min_value=0.0, max_value=1.0, value=None, placeholder="Leave blank for N/A", step=0.01)

                st.markdown("---")
                st.markdown("### EXPANDED OIL ANALYSIS (IEC 60422 PARAMETERS)")

                o1, o2_col, o3, o4 = st.columns(4)
                bdv = o1.number_input("BDV (kV)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                acid = o2_col.number_input("Acid Number (mgKOH/g)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.01)
                water = o3.number_input("Water Content (ppm)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)
                ift = o4.number_input("IFT (mN/m)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)

                o5, o6, o7, o8 = st.columns(4)
                ddf = o5.number_input("DDF / Tan Delta", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.001)
                resistivity = o6.number_input("Resistivity (Gohm.m)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=1.0)
                colour_iso = o7.number_input("Oil Colour (ISO)", min_value=0.5, max_value=8.0, value=None, placeholder="Leave blank for N/A", step=0.5)
                sediment_sludge = o8.selectbox("Sediment / Sludge State", ["Inherit Last", "No", "Sediment", "Sludge"])

                o9, o10, o11, o12 = st.columns(4)
                corrosive_sulfur = o9.selectbox("Corrosive Sulphur", ["Inherit Last", "Non-Corrosive", "Corrosive"])
                particles_iso = o10.selectbox("Particles ISO 4406", ["Inherit Last", "Good", "Fair", "Poor"])
                inhibitor = o11.number_input("Inhibitor Content (%)", min_value=0.0, max_value=100.0, value=None, placeholder="Leave blank for N/A", step=1.0)
                passivator = o12.number_input("Passivator (mg/kg)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=1.0)

                c_f1, c_f2 = st.columns(2)
                flash_point = c_f1.number_input("Flash Point (°C)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=1.0)
                pcb_content = c_f2.number_input("PCB Content (mg/kg)", min_value=0.0, value=None, placeholder="Leave blank for N/A", step=0.1)

                if st.form_submit_button("EXECUTE EVALUATION & SAVE RECORD"):
                    input_dict = {
                        'ID_Trafo': trafo_id_input.strip(),
                        'Tanggal_Uji': str(test_date),
                        'H2': float(h2) if h2 is not None else None,
                        'CH4': float(ch4) if ch4 is not None else None,
                        'C2H6': float(c2h6) if c2h6 is not None else None,
                        'C2H4': float(c2h4) if c2h4 is not None else None,
                        'C2H2': float(c2h2) if c2h2 is not None else None,
                        'CO': float(co) if co is not None else None,
                        'CO2': float(co2) if co2 is not None else None,
                        'Ratio_O2_N2': float(ratio_o2_n2) if ratio_o2_n2 is not None else None,
                        'BDV': float(bdv) if bdv is not None else None,
                        'Acid': float(acid) if acid is not None else None,
                        'Water': float(water) if water is not None else None,
                        'IFT': float(ift) if ift is not None else None,
                        'DDF': float(ddf) if ddf is not None else None,
                        'Resistivity': float(resistivity) if resistivity is not None else None,
                        'Colour_ISO2049': float(colour_iso) if colour_iso is not None else None,
                        'Inhibitor_Content': float(inhibitor) if inhibitor is not None else None,
                        'Passivator_Content': float(passivator) if passivator is not None else None,
                        'Flash_Point': float(flash_point) if flash_point is not None else None,
                        'PCB_Content': float(pcb_content) if pcb_content is not None else None,
                        'Sediment_Sludge': sediment_sludge if sediment_sludge != "Inherit Last" else None,
                        'Corrosive_Sulphur': corrosive_sulfur if corrosive_sulfur != "Inherit Last" else None,
                        'Particles_ISO': particles_iso if particles_iso != "Inherit Last" else None,
                        'Status_Pemurnian': purification_status,
                        'Is_Anomali': 'No'
                    }

                    has_anomaly, anomaly_reason = check_anomaly(df_all, trafo_id_input.strip(), pd.to_datetime(test_date), input_dict)
                    if has_anomaly:
                        st.session_state['pending_data'] = input_dict
                        st.session_state['anomaly_reason'] = anomaly_reason
                        st.rerun()
                    else:
                        insert_data(input_dict)
                        st.success(f"Record for '{trafo_id_input.strip()}' saved successfully.")
                        st.rerun()

with tab_data:
    ui.render_title("MASTER LEDGER & RECORD MANAGEMENT")
    if not df_all.empty:
        with st.spinner("Processing analytical models..."):
            df_prognosis = calculate_prognosis_and_prediction(df_all)
        st.dataframe(df_prognosis, use_container_width=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            buf1 = io.BytesIO()
            with pd.ExcelWriter(buf1, engine='openpyxl') as w: df_prognosis.to_excel(w, index=False)
            st.download_button("EXPORT PROGNOSIS LEDGER (.XLSX)", buf1.getvalue(), f"Prognosis_{datetime.now().strftime('%Y%m%d')}.xlsx")
        with col_exp2:
            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine='openpyxl') as w: df_metadata.to_excel(w, index=False)
            st.download_button("EXPORT NAMEPLATE METADATA (.XLSX)", buf2.getvalue(), f"Nameplates_{datetime.now().strftime('%Y%m%d')}.xlsx")

        st.markdown("---")
        ui.render_title("HISTORICAL RECORD DELETION CONTROL")
        delete_options = df_all.apply(lambda x: f"ID: {x['id']} | Transformer: {x['ID_Trafo']} | Date: {x['Tanggal_Uji']}", axis=1).tolist()
        record_to_delete = st.selectbox("SELECT HISTORICAL RECORD TO PURGE", ["-- Select Record --"] + delete_options)

        if st.button("PURGE SELECTED HISTORICAL ROW"):
            if record_to_delete != "-- Select Record --":
                db_id = int(record_to_delete.split("|")[0].replace("ID:", "").strip())
                st.session_state['delete_confirm_id'] = db_id

        if 'delete_confirm_id' in st.session_state and st.session_state['delete_confirm_id'] is not None:
            st.warning("CONFIRMATION: Are you sure you want to purge this record?")
            col_d1, col_d2 = st.columns(2)
            if col_d1.button("CONFIRM PURGE"):
                delete_data(st.session_state['delete_confirm_id'])
                st.session_state['delete_confirm_id'] = None
                st.rerun()
            if col_d2.button("CANCEL"):
                st.session_state['delete_confirm_id'] = None
                st.rerun()

with tab_insights:
    ui.render_title("DIAGNOSTIC INSIGHTS & TECHNICAL DATASHEET")
    if not df_all.empty:
        df_prog_insight = calculate_prognosis_and_prediction(df_all)
        insight_trafo = st.selectbox("SELECT TRANSFORMER UNIT", df_prog_insight['ID_Trafo'].unique())
        df_insight_filtered = df_prog_insight[(df_prog_insight['ID_Trafo'] == insight_trafo) & (df_prog_insight['Tipe_Data'] == 'Historical')].sort_values('Tanggal_Uji').reset_index(drop=True)

        if not df_insight_filtered.empty:
            latest_record = df_insight_filtered.iloc[-1]
            meta_match = df_metadata[df_metadata['ID_Trafo'] == insight_trafo]
            
            if not meta_match.empty and pd.notna(meta_match.iloc[0].get('Year_Manufactured')):
                y_manuf = meta_match.iloc[0]['Year_Manufactured']
                curr_year = pd.to_datetime(latest_record['Tanggal_Uji']).year
                disp_age = f"{curr_year - y_manuf} Years (Mfg: {y_manuf})"
                disp_manuf = meta_match.iloc[0].get('Manufacturer', 'N/A')
                disp_cap = f"{meta_match.iloc[0].get('Capacity_MVA', 'N/A')} MVA"
            else:
                disp_age, disp_manuf, disp_cap = "Unknown", "N/A", "N/A"

            fault_rows = df_insight_filtered[df_insight_filtered['Status_DGA'] != 'Normal']
            first_fault_text = f"{fault_rows.iloc[0]['Tanggal_Uji']} ({fault_rows.iloc[0]['Status_DGA']})" if not fault_rows.empty else "NONE DETECTED (Normal Status)"

            ui.render_insights_datasheet(latest_record, disp_manuf, disp_cap, disp_age, first_fault_text)

with tab_trend:
    ui.render_title("HMI GAS DYNAMICS & INSTRUMENT GAUGES")
    if df_all.empty:
        st.info("No test data available for the selected transformer.")
    else:
        trafo_filter = st.selectbox("SELECT TRANSFORMER UNIT", df_all['ID_Trafo'].unique(), key="trend_selector")
        df_graph = calculate_prognosis_and_prediction(df_all)
        df_filtered = df_graph[df_graph['ID_Trafo'] == trafo_filter].sort_values('Tanggal_Uji').reset_index(drop=True)

        fig_gas = go.Figure()
        gas_list, colors = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2'], ['#38BDF8', '#F59E0B', '#10B981', '#EF4444', '#A855F7']
        df_hist_only = df_filtered[df_filtered['Tipe_Data'] == 'Historical']
        last_hist_date = df_hist_only['Tanggal_Uji'].iloc[-1] if not df_hist_only.empty else None

        for idx_g, gas in enumerate(gas_list):
            if gas in df_filtered.columns:
                fig_gas.add_trace(go.Scatter(x=df_filtered['Tanggal_Uji'], y=df_filtered[gas], mode='lines+markers', name=gas, line=dict(color=colors[idx_g], width=2)))

        if last_hist_date:
            fig_gas.add_vline(x=last_hist_date, line_width=1.5, line_dash="dash", line_color="#D97706")

        fig_gas.update_layout(
            title=dict(text=f"DGA GAS EVOLUTION & FORECAST - {trafo_filter}", font=dict(family="IBM Plex Mono", color="#0F172A", size=14)),
            xaxis_title="TEST DATE", yaxis_title="CONCENTRATION (PPM)", template="plotly_white",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=420, margin=dict(l=40, r=40, t=50, b=40)
        )

        st.plotly_chart(fig_gas, use_container_width=True)

        ui.render_title("CURRENT DGA GAS CONCENTRATIONS (EXCLUDING O2/N2 RATIO)")
        if not df_hist_only.empty:
            ui.render_gas_ledger(df_hist_only.iloc[-1])

        ui.render_title("CURRENT EXTENDED OIL ANALYSIS (OA) CATEGORICAL STATUS LEDGER")
        if not df_hist_only.empty:
            ui.render_oa_ledger(df_hist_only.iloc[-1])

        ui.render_title("PHYSICAL & CHEMICAL OIL GAUGES (LAST OBSERVATION - CATEGORY C <72.5 kV)")
        if not df_hist_only.empty:
            last_oa = df_hist_only.iloc[-1]

            g1, g2, g3, g4 = st.columns(4)
            g1.plotly_chart(create_hmi_gauge(float(last_oa.get('BDV', 0)) if pd.notna(last_oa.get('BDV')) else 0.0, "BDV (kV) [Min 40]", 0, 100, [{'range': [0, 30], 'color': "#DC2626"}, {'range': [30, 40], 'color': "#D97706"}, {'range': [40, 100], 'color': "#16A34A"}]), use_container_width=True)
            g2.plotly_chart(create_hmi_gauge(float(last_oa.get('Acid', 0)) if pd.notna(last_oa.get('Acid')) else 0.0, "Acid (mgKOH/g) [Max 0.15]", 0, 0.5, [{'range': [0, 0.15], 'color': "#16A34A"}, {'range': [0.15, 0.30], 'color': "#D97706"}, {'range': [0.30, 0.5], 'color': "#DC2626"}]), use_container_width=True)
            g3.plotly_chart(create_hmi_gauge(float(last_oa.get('Water', 0)) if pd.notna(last_oa.get('Water')) else 0.0, "Water (ppm) [Max 30]", 0, 60, [{'range': [0, 30], 'color': "#16A34A"}, {'range': [30, 40], 'color': "#D97706"}, {'range': [40, 60], 'color': "#DC2626"}]), use_container_width=True)
            g4.plotly_chart(create_hmi_gauge(float(last_oa.get('IFT', 0)) if pd.notna(last_oa.get('IFT')) else 0.0, "IFT (mN/m) [Min 28]", 0, 50, [{'range': [0, 22], 'color': "#DC2626"}, {'range': [22, 28], 'color': "#D97706"}, {'range': [28, 50], 'color': "#16A34A"}]), use_container_width=True)

            g5, g6, g7, g8 = st.columns(4)
            g5.plotly_chart(create_hmi_gauge(float(last_oa.get('DDF', 0)) if pd.notna(last_oa.get('DDF')) else 0.0, "DDF/Tan Delta [Max 0.10]", 0, 0.60, [{'range': [0, 0.10], 'color': "#16A34A"}, {'range': [0.10, 0.50], 'color': "#D97706"}, {'range': [0.50, 0.60], 'color': "#DC2626"}]), use_container_width=True)
            g6.plotly_chart(create_hmi_gauge(float(last_oa.get('Resistivity', 0)) if pd.notna(last_oa.get('Resistivity')) else 0.0, "Resistivity Gohm.m [Min 60]", 0, 100, [{'range': [0, 4], 'color': "#DC2626"}, {'range': [4, 60], 'color': "#D97706"}, {'range': [60, 100], 'color': "#16A34A"}]), use_container_width=True)
            g7.plotly_chart(create_hmi_gauge(float(last_oa.get('Colour_ISO2049', 0.5)) if pd.notna(last_oa.get('Colour_ISO2049')) else 0.5, "Colour (ISO 2049) [Max 2.0]", 0.5, 8.0, [{'range': [0.5, 2.0], 'color': "#16A34A"}, {'range': [2.0, 3.5], 'color': "#D97706"}, {'range': [3.5, 8.0], 'color': "#DC2626"}]), use_container_width=True)
            g8.plotly_chart(create_hmi_gauge(float(last_oa.get('Inhibitor_Content', 0)) if pd.notna(last_oa.get('Inhibitor_Content')) else 0.0, "Inhibitor Content (%) [Min 60]", 0, 100, [{'range': [0, 40], 'color': "#DC2626"}, {'range': [40, 60], 'color': "#D97706"}, {'range': [60, 100], 'color': "#16A34A"}]), use_container_width=True)

            g9, g10, g11, _ = st.columns(4)
            g9.plotly_chart(create_hmi_gauge(float(last_oa.get('Passivator_Content', 0)) if pd.notna(last_oa.get('Passivator_Content')) else 0.0, "Passivator (mg/kg) [Min 70]", 0, 150, [{'range': [0, 50], 'color': "#DC2626"}, {'range': [50, 70], 'color': "#D97706"}, {'range': [70, 150], 'color': "#16A34A"}]), use_container_width=True)
            g10.plotly_chart(create_hmi_gauge(float(last_oa.get('Flash_Point', 0)) if pd.notna(last_oa.get('Flash_Point')) else 0.0, "Flash Point (°C) [Min 135]", 0, 180, [{'range': [0, 120], 'color': "#DC2626"}, {'range': [120, 135], 'color': "#D97706"}, {'range': [135, 180], 'color': "#16A34A"}]), use_container_width=True)
            g11.plotly_chart(create_hmi_gauge(float(last_oa.get('PCB_Content', 0)) if pd.notna(last_oa.get('PCB_Content')) else 0.0, "PCB Content (mg/kg) [Max 2.0]", 0, 10, [{'range': [0, 2.0], 'color': "#16A34A"}, {'range': [2.0, 5.0], 'color': "#D97706"}, {'range': [5.0, 10.0], 'color': "#DC2626"}]), use_container_width=True)

with tab_duval:
    ui.render_title("DUVAL TRIANGLE 1 DIAGNOSTIC PANEL (IEEE C57.104 / IEC 60599)")

    if df_all.empty:
        st.info("No test data available for the selected transformer.")
    else:
        df_duval = calculate_prognosis_and_prediction(df_all)
        duval_trafo = st.selectbox("SELECT TRANSFORMER UNIT", df_duval['ID_Trafo'].unique(), key="duval_selector")

        df_d_filtered = df_duval[(df_duval['ID_Trafo'] == duval_trafo) & (df_duval['Tipe_Data'] == 'Historical')].sort_values('Tanggal_Uji').reset_index(drop=True)

        if not df_d_filtered.empty:
            latest_d = df_d_filtered.iloc[-1]
            current_ieee_status = str(latest_d.get('DGA_Status_IEEE', '')).strip()

            if current_ieee_status == "Status 1":
                st.info(f"**DUVAL TRIANGLE INACTIVE FOR {duval_trafo}**\n\n"
                        f"According to IEEE C57.104 standards, Duval Triangle analysis is only applicable when gas concentrations exceed normal operating thresholds.\n\n"
                        f"- **Latest IEEE Status ({latest_d.get('Tanggal_Uji')}):** `{current_ieee_status}` (Normal Condition)\n"
                        f"- **Physics Fault Verdict:** `{latest_d.get('Status_DGA')}`\n\n"
                        f"*The Duval Triangle plot will automatically render if the transformer condition escalates to Status 2 (Caution) or Status 3 (Fault).*")
            else:
                c_ch4 = float(latest_d.get('CH4', 0)) if pd.notna(latest_d.get('CH4')) else 0.0
                c_c2h4 = float(latest_d.get('C2H4', 0)) if pd.notna(latest_d.get('C2H4')) else 0.0
                c_c2h2 = float(latest_d.get('C2H2', 0)) if pd.notna(latest_d.get('C2H2')) else 0.0

                fig_duval, p_ch4, p_c2h4, p_c2h2 = plot_duval_triangle1(c_ch4, c_c2h4, c_c2h2, duval_trafo)

                col_d_left, col_d_right = st.columns([2, 1])

                with col_d_left:
                    st.plotly_chart(fig_duval, use_container_width=True)

                with col_d_right:
                    st.markdown("### CURRENT GAS PROPORTIONS")
                    st.write(f"**Observation Date:** `{latest_d.get('Tanggal_Uji')}`")
                    st.write(f"**IEEE Status:** `{current_ieee_status}`")
                    st.write(f"**% CH4:** `{p_ch4:.2f} %`")
                    st.write(f"**% C2H4:** `{p_c2h4:.2f} %`")
                    st.write(f"**% C2H2:** `{p_c2h2:.2f} %`")
                    st.markdown("---")
                    st.write(f"**Physics Fault Verdict:** `{latest_d.get('Status_DGA')}`")
                    st.write(f"**Paper Status:** `{latest_d.get('Paper_Status')}`")
