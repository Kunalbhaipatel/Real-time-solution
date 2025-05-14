import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import base64
from fpdf import FPDF
import openai

st.set_page_config(layout="wide")
st.title("🛠️ Drilling Operations Monitoring Dashboard")

# --- Upload Section ---
st.sidebar.header("Upload Drilling Sensor CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# --- GPT Natural Language Summary ---
def generate_summary(alerts, recommendations):
    if not alerts and not recommendations:
        return "No anomalies detected during this session. Operations appear stable."

    prompt = f"""
    Based on the following drilling anomalies and expert recommendations, provide a concise summary suitable for rig supervisors and mud engineers:

    Alerts:
    {chr(10).join(['- ' + a for a in alerts])}

    Recommendations:
    {chr(10).join(['- ' + r for r in recommendations])}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# --- PDF Report Generator ---
def generate_pdf_report(alerts, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Drilling Operations Alert Summary", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Alerts", ln=True)
    pdf.set_font("Arial", size=12)
    for alert in alerts:
        pdf.multi_cell(0, 10, f"- {alert}")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Recommendations", ln=True)
    pdf.set_font("Arial", size=12)
    for rec in recommendations:
        pdf.multi_cell(0, 10, f"- {rec}")

    return pdf.output(dest='S').encode('latin-1')

# Continue only if file uploaded
if uploaded_file:
    with st.spinner("Processing uploaded data..."):
        usecols = [
            'YYYY/MM/DD', 'HH:MM:SS',
            'Rate Of Penetration (ft_per_hr)', 'PLC ROP (ft_per_hr)',
            'Hook Load (klbs)', 'Standpipe Pressure (psi)',
            'Pump 1 strokes/min (SPM)', 'Pump 2 strokes/min (SPM)',
            'DAS Vibe Lateral Max (g_force)', 'DAS Vibe Axial Max (g_force)',
            'AutoDriller Limiting (unitless)',
            'DAS Vibe WOB Reduce (percent)', 'DAS Vibe RPM Reduce (percent)'
        ]

        df = pd.read_csv(uploaded_file, usecols=usecols)
        df['Timestamp'] = pd.to_datetime(df['YYYY/MM/DD'] + ' ' + df['HH:MM:SS'], format='%m/%d/%Y %H:%M:%S')
        df.set_index('Timestamp', inplace=True)
        df.drop(columns=['YYYY/MM/DD', 'HH:MM:SS'], inplace=True)

        st.success("Data loaded successfully!")
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df.head(10))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Monitoring Dashboard", "🚨 Alerts & Expert Guidance", "🧾 GPT Summary", "📊 Statistical Analysis", "🔬 ML Detections"])

        with tab1:
            st.markdown("### Real-Time Sensor Trends")
            for col in df.columns:
                fig = px.line(df, x=df.index, y=col, title=col)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 🚨 Detected Alerts")
            alerts = []

            df['ROP_change'] = df['Rate Of Penetration (ft_per_hr)'].pct_change().abs()
            if df['ROP_change'].rolling('10min').mean().gt(0.5).any():
                alerts.append("Significant ROP fluctuation (>50% in 10 min) detected.")

            hl_issue = (df['Hook Load (klbs)'] > 60) &                        (df['Rate Of Penetration (ft_per_hr)'] < 1) &                        (df['Pump 1 strokes/min (SPM)'] == 0)
            if hl_issue.any():
                alerts.append("High hook load while pumps are off and ROP is near zero. Possible stuck pipe.")

            if df['DAS Vibe Lateral Max (g_force)'].gt(25).any():
                alerts.append("Excessive lateral vibration detected (>25g).")

            if df['AutoDriller Limiting (unitless)'].gt(0).any():
                alerts.append("AutoDriller limiting detected.")

            if df['DAS Vibe WOB Reduce (percent)'].gt(0).any() or df['DAS Vibe RPM Reduce (percent)'].gt(0).any():
                alerts.append("DAS vibration mitigation active.")

            if alerts:
                for alert in alerts:
                    st.error(alert)
            else:
                st.success("No critical alerts detected.")

            st.markdown("### 🧠 Expert Recommendations")
            recommendations = [
                "Check mud properties for signs of cuttings buildup or influx.",
                "Consider circulating bottoms-up if ROP drops are persistent.",
                "Inspect for stuck pipe conditions if hook load rises during no-ROP periods.",
                "Tune RPM and WOB if vibration levels exceed 25g.",
                "Monitor shaker screen loading if flow and ROP fluctuate rapidly."
            ]
            for rec in recommendations:
                st.markdown(f"- {rec}")

        with tab3:
            st.markdown("### 🤖 GPT-Based Summary")
            summary = generate_summary(alerts, recommendations)
            st.text_area("Session Summary", summary, height=300)

        with tab4:
            st.markdown("### 📊 Statistical Summary of Key Metrics")
            stats = df.describe().T[['mean', 'std', 'min', 'max']]
            st.dataframe(stats)

            st.markdown("### 📥 Download Statistical Summary")
            stat_csv = stats.to_csv().encode('utf-8')
            st.download_button("Download Statistics CSV", stat_csv, "statistical_summary.csv")

        with tab5:
            st.markdown("### 🔬 Machine Learning-Based Detections")
            import joblib
            try:
                model = joblib.load("model.pkl")
                features = ['PLC ROP (ft_per_hr)', 'DAS Vibe Lateral Max (g_force)']
                df = df.dropna(subset=features)
                X = df[features]
                df['ML Screen Efficiency Flag'] = model.predict(X)
            except Exception as e:
                st.warning(f"⚠️ Could not load or run ML model: {e}")
                df['ML Screen Efficiency Flag'] = 0

            st.write("Flag = 1 indicates predicted screen overload risk.")
            st.dataframe(df[['PLC ROP (ft_per_hr)', 'DAS Vibe Lateral Max (g_force)', 'ML Screen Efficiency Flag']].tail(50))

            st.markdown("### 🧭 ML Flag Timeline")
            fig = px.scatter(
                df.reset_index(),
                x='Timestamp',
                y='ML Screen Efficiency Flag',
                color='ML Screen Efficiency Flag',
                title='Screen Overload Risk Flag Over Time',
                labels={"ML Screen Efficiency Flag": "Flag (1 = Risk)"},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📥 Download ML Flags")
            ml_export = df[['PLC ROP (ft_per_hr)', 'DAS Vibe Lateral Max (g_force)', 'ML Screen Efficiency Flag']]
            st.download_button("Download ML Flags CSV", ml_export.to_csv().encode('utf-8'), "ml_screen_efficiency_flags.csv")

        st.sidebar.markdown("---")
        st.sidebar.header("📤 Export")
        csv = df.to_csv().encode('utf-8')
        st.sidebar.download_button("Download Cleaned CSV", csv, "processed_data.csv")

        if alerts:
            pdf_bytes = generate_pdf_report(alerts, recommendations)
            st.sidebar.download_button("Download PDF Report", data=pdf_bytes, file_name="drilling_alerts_summary.pdf")
