import os
import pandas as pd
import streamlit as st

def load_css(file_name="style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header(title_text):
    st.markdown(f"<div class='sap-header'>{title_text}</div>", unsafe_allow_html=True)

def render_title(title_text):
    st.markdown(f"<div class='sap-title'>{title_text}</div>", unsafe_allow_html=True)

def fmt_val(v, unit="ppm"):
    return f"{float(v):.1f} {unit}" if pd.notna(v) else f"0.0 {unit}"

def render_insights_datasheet(latest_record, disp_manuf, disp_cap, disp_age, first_fault_text):
    html = f"""
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
    """
    st.markdown(html, unsafe_allow_html=True)

def render_gas_ledger(last_row):
    html = f"""
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
    """
    st.markdown(html, unsafe_allow_html=True)

def render_oa_ledger(last_row):
    html = f"""
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
    """
    st.markdown(html, unsafe_allow_html=True)