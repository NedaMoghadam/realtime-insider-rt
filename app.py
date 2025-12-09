import os
import tempfile
from typing import List, Tuple

import pandas as pd
import streamlit as st

from pipeline_rt import run_realtime_detection  # your existing pipeline


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
# Streamlit page config + main header
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Insider Threat Detection – Neda B. Moghadam",
    layout="wide",
)

st.title("Real-Time Insider Threat Detection with TinyLlama + Early Exit")
st.markdown("**Developed by: Neda B. Moghadam (Polytechnique Montréal)**")
st.markdown("---")

st.markdown(
    """
Upload one or **more** log CSV file(s) (CERT, LANL, or your own enterprise logs).  
They will be concatenated, converted to daily user activity,  
and anomalies will appear **in real time** as sliding windows are processed.
"""
)

# -------------------------------------------------------------------
# Sidebar controls (with About box)
# -------------------------------------------------------------------
st.sidebar.header("Settings")

st.sidebar.title("About this app")
st.sidebar.info(
    "Real-time insider threat detection demo using TinyLlama and early-exit, "
    "developed by **Neda B. Moghadam**.\n\n"
    "Upload your own log CSV file(s) and run the full pipeline in real time."
)

test_mode = st.sidebar.checkbox("Test mode (first 1000 rows per file only)", value=True)
window_min = st.sidebar.slider("Window size (minutes)", 10, 180, 60, 10)
slide_min = st.sidebar.slider("Slide size (minutes)", 1, 60, 10, 1)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def merge_files_for_analysis(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    test_mode: bool = True,
) -> Tuple[str, int]:
    """
    Read one or more uploaded CSVs into a single temporary CSV file.

    Returns:
        merged_path: path to temporary merged file
        total_rows:  total number of rows used
    """
    dfs = []
    total_rows = 0

    for f in uploaded_files:
        # f is a Streamlit UploadedFile
        df = pd.read_csv(f, low_memory=False)

        if test_mode:
            df = df.head(1000)

        dfs.append(df)
        total_rows += len(df)

    if not dfs:
        raise ValueError("No data in uploaded files.")

    merged_df = pd.concat(dfs, ignore_index=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        merged_df.to_csv(tmp.name, index=False)
        merged_path = tmp.name

    return merged_path, total_rows


# -------------------------------------------------------------------
# Main layout placeholders
# -------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "📁 Upload your log CSV file(s)", type=["csv"], accept_multiple_files=True
)
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
    """
    Called from pipeline_rt for every processed time window.

    It updates:
      - progress bar
      - textual log
      - live charts (events vs anomalies, anomaly rate)
    """
    # Be robust to different metric names: "events" or "n_events"
    events = info.get("events", info.get("n_events", 0)) or 0
    anomalies = info.get("anomalies", info.get("n_anomalies", 0)) or 0

    # Ensure they are ints
    events = int(events)
    anomalies = int(anomalies)

    # Progress bar
    progress = (slot_idx + 1) / max(total_slots, 1)
    progress_bar.progress(progress)

    # Status line
    status_text.info(
        f"Processing window {slot_idx+1}/{total_slots} | "
        f"{info['timestamp']} → {events} events, {anomalies} anomalies"
    )

    # Text log
    with log_container:
        st.write(
            f"Slot {slot_idx+1:3d} | {info['timestamp']} | "
            f"{events:4d} events → **{anomalies:2d} anomalies**"
        )

    # Live chart data update
    anomaly_rate = anomalies / events if events > 0 else 0.0

    st.session_state["live_chart_data"].append(
        {
            "timestamp": info["timestamp"],
            "events": events,
            "anomalies": anomalies,
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
if uploaded_files and run_button:
    # Reset live charts for a fresh run
    st.session_state["live_chart_data"] = []

    # 1) Merge all uploaded CSVs into a single temporary file
    merged_path, total_rows = merge_files_for_analysis(uploaded_files, test_mode=test_mode)

    st.success(
        f"{len(uploaded_files)} file(s) merged → {total_rows:,} rows. "
        "Starting real-time analysis..."
    )

    try:
        # 2) Call your pipeline with the **correct signature**
        anomalies_df, output_folder = run_realtime_detection(
            file_path=merged_path,
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
                # Basic table
                display_cols = [
                    col
                    for col in ["timestamp", "user", "pc", "activity", "score"]
                    if col in anomalies_df.columns
                ]
                st.dataframe(
                    anomalies_df[display_cols].head(100),
                    use_container_width=True,
                )

                # ---------------- Extra visualisations ----------------
                # 1) Anomalies per user
                if "user" in anomalies_df.columns:
                    st.subheader("👤 Anomalies per User (top 20)")
                    user_counts = anomalies_df["user"].value_counts().head(20)
                    st.bar_chart(user_counts)

                # 2) Score distribution for anomalies
                if "score" in anomalies_df.columns:
                    st.subheader("📈 Anomaly Score Distribution")
                    st.histogram(anomalies_df["score"])

                # 3) Anomalies over time (if timestamp available)
                if "timestamp" in anomalies_df.columns:
                    st.subheader("⏱ Anomalies Over Time")
                    time_counts = (
                        anomalies_df.set_index("timestamp")
                        .resample("1H")
                        .size()
                        .rename("n_anomalies")
                    )
                    st.line_chart(time_counts)

            else:
                st.info("No anomalies found.")

            st.subheader("📂 Output folder")
            st.code(str(output_folder))

            plot_path = os.path.join(output_folder, "score_distribution.png")
            if os.path.exists(plot_path):
                st.image(
                    plot_path,
                    caption="Anomaly Score Distribution (pipeline plot)",
                    use_column_width=True,
                )

    except Exception as e:
        st.error(f"Error during analysis: {e}")

    finally:
        if os.path.exists(merged_path):
            try:
                os.unlink(merged_path)
            except Exception:
                pass

# -------------------------------------------------------------------
# Footer with name
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "© 2025 **Neda B. Moghadam** · Real-Time Insider Threat Detection Demo "
    "· For research and educational use."
)
