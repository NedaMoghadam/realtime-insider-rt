import os
import tempfile
import pandas as pd
import streamlit as st

from pipeline_rt import run_realtime_detection  # uses your existing pipeline_rt.py

# -------------------------------------------------------------------
# Output directory (use your existing base folder)
# -------------------------------------------------------------------
try:
    from config import OUTPUT_BASE_DIR
    OUTPUT_DIR = OUTPUT_BASE_DIR
except Exception:
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Insider Threat Detection",
    layout="wide",
)

st.title("Real-Time Insider Threat Detection with TinyLlama + Early Exit")

st.markdown(
    """
Upload a **log CSV file** (CERT, LANL, or your own enterprise logs)  
and watch anomalies appear **in real time** as sliding windows are processed.
"""
)

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Settings")

test_mode = st.sidebar.checkbox("Test mode (first 1000 rows only)", value=True)

window_min = st.sidebar.slider("Window size (minutes)", 10, 180, 60, 10)
slide_min = st.sidebar.slider("Slide size (minutes)", 1, 60, 10, 1)

# -------------------------------------------------------------------
# Main layout placeholders
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your log CSV file", type=["csv"])
run_button = st.button("🚀 Start Real-Time Detection", type="primary")

status_text = st.empty()
progress_bar = st.progress(0)

log_container = st.container()
charts_container = st.container()
results_container = st.container()

if "live_chart_data" not in st.session_state:
    st.session_state["live_chart_data"] = []


# -------------------------------------------------------------------
# Callback used by the pipeline for each time window
# -------------------------------------------------------------------
def live_callback(slot_idx, total_slots, info):

    progress = (slot_idx + 1) / max(total_slots, 1)
    progress_bar.progress(progress)

    status_text.info(
        f"Processing window {slot_idx+1}/{total_slots} | "
        f"{info['timestamp']} → {info['events']} events, "
        f"{info['anomalies']} anomalies"
    )

    with log_container:
        st.write(
            f"Slot {slot_idx+1:3d} | {info['timestamp']} | "
            f"{info['events']:4d} events → **{info['anomalies']:2d} anomalies**"
        )

    # Live chart data update
    anomaly_rate = info["anomalies"] / info["events"] if info["events"] > 0 else 0.0

    st.session_state["live_chart_data"].append(
        {
            "timestamp": info["timestamp"],
            "events": info["events"],
            "anomalies": info["anomalies"],
            "anomaly_rate": anomaly_rate,
        }
    )

    chart_df = pd.DataFrame(st.session_state["live_chart_data"])

    if not chart_df.empty:
        chart_df = chart_df.drop_duplicates(subset=["timestamp"])
        chart_df = chart_df.sort_values("timestamp")
        chart_df = chart_df.set_index("timestamp")

        with charts_container:
            st.subheader("📊 Live Window Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Events vs Anomalies per Window")
                st.line_chart(chart_df[["events", "anomalies"]])

            with col2:
                st.caption("Anomaly Rate per Window")
                st.line_chart(chart_df[["anomaly_rate"]])


# -------------------------------------------------------------------
# Main run
# -------------------------------------------------------------------
if uploaded_file and run_button:

    st.session_state["live_chart_data"] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.success("File uploaded — starting real-time analysis...")

    try:
        anomalies_df, output_folder = run_realtime_detection(
            file_path=tmp_path,
            output_base_dir=OUTPUT_DIR,
            test_mode=test_mode,
            window_time=window_min,
            slide_time=slide_min,
            status_callback=live_callback,
        )

        st.success("✅ Analysis completed!")
        progress_bar.progress(1.0)

        with results_container:
            st.subheader("🔎 Top Detected Anomalies")

            if anomalies_df is not None and not anomalies_df.empty:
                display_cols = [
                    col for col in ["timestamp", "user", "pc", "activity", "score"]
                    if col in anomalies_df.columns
                ]
                st.dataframe(anomalies_df[display_cols].head(50), use_container_width=True)
            else:
                st.info("No anomalies found.")

            st.subheader("📂 Output folder")
            st.code(str(output_folder))

            plot_path = os.path.join(output_folder, "score_distribution.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Anomaly Score Distribution", use_column_width=True)

    except Exception as e:
        st.error(f"Error during analysis: {e}")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
