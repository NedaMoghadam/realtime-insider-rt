"""
Real-Time Detection Pipeline — Final Stable Version (A + B + C + Safe Index Fix)
Works with:
- config.py
- logger.py
- exceptions.py
- data_loader.py
- tinyllama_model.py
- anomaly_detector.py
"""

import os
import time
from datetime import datetime
from typing import Callable, Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from logger import get_logger
from data_loader import DataLoader
from tinyllama_model import TinyLlamaModel
from anomaly_detector import AnomalyDetector

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Convert row → TinyLlama text
# ----------------------------------------------------------------------
def render_event(row: pd.Series) -> str:
    ts = row["date"].strftime("%Y-%m-%d %H:%M")
    return (
        f"[{ts}] user={row['user']} role={row['role']} pc={row['pc']} "
        f"activity={row['activity']} hour={row['hour']} dow={row['day_of_week']}"
    )


# ======================================================================
# REAL-TIME PIPELINE (BEST VERSION)
# ======================================================================
class RealtimePipeline:
    """
    Combined improvements:
    - Robust preprocessing
    - Deterministic namespaced user/pc IDs
    - EMA user embeddings
    - Rolling baseline IsolationForest
    - Safe indexing (no out-of-bounds crash)
    - Expanded feature vector
    """

    def __init__(self):
        logger.info("=" * 70)
        logger.info("Initializing Real-Time Combined Pipeline")
        logger.info("=" * 70)

        self.loader = DataLoader(
            required_columns=config.REQUIRED_COLUMNS,
            optional_columns=config.OPTIONAL_COLUMNS,
            date_column="date",
            date_format=config.DATE_FORMAT,
        )

        # TinyLlama model
        self.model = TinyLlamaModel(
            model_id=config.MODEL_ID,
            cache_dir=None,
            use_gpu=config.USE_GPU,
            max_length=config.MAX_SEQUENCE_LENGTH,
        )

        # Rolling baseline anomaly detector
        self.detector = AnomalyDetector(
            contamination=config.CONTAMINATION_FACTOR,
            random_state=42,
        )

        # Rolling baseline memory
        self.baseline_features: List[np.ndarray] = []

        self.hidden_size = self.model.hidden_size

        logger.info("Pipeline initialized.")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------
    def run(
        self,
        input_file: str,
        output_dir: str,
        test_mode: bool = True,
        window_time: int = 60,
        slide_time: int = 10,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> Dict:

        os.makedirs(output_dir, exist_ok=True)

        nrows = config.TEST_MODE_ROWS if test_mode else None
        df = self.loader.load(input_file, nrows=nrows)

        df = self._preprocess(df)

        results = self._process_windows(
            df,
            window_time=window_time,
            slide_time=slide_time,
            callback=callback,
        )

        self._save_results(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # STEP 1 — PREPROCESSING
    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing...")

        df = df.copy()
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek

        # Fill NaNs
        df["activity"] = df["activity"].fillna("UNKNOWN_ACTIVITY").astype(str)
        df["role"]     = df["role"].fillna("UNKNOWN_ROLE").astype(str)
        df["user"]     = df["user"].fillna("UNKNOWN_USER").astype(str)
        df["pc"]       = df["pc"].fillna("UNKNOWN_PC").astype(str)

        # Encoders
        df["activity_encoded"] = LabelEncoder().fit_transform(df["activity"])
        df["role_encoded"]     = LabelEncoder().fit_transform(df["role"])

        # Deterministic namespaced IDs
        user_keys = [f"user:{u}" for u in df["user"]]
        pc_keys   = [f"pc:{p}"   for p in df["pc"]]

        unique_nodes = sorted(set(user_keys + pc_keys))
        node_map = {node: i for i, node in enumerate(unique_nodes)}

        df["user_id"] = [node_map[k] for k in user_keys]
        df["pc_id"]   = [node_map[k] for k in pc_keys]

        logger.info(f"Preprocessing complete: {len(df)} events")
        return df

    # ------------------------------------------------------------------
    # STEP 2 — SLIDING WINDOWS
    # ------------------------------------------------------------------
    def _process_windows(
        self, df: pd.DataFrame, window_time: int, slide_time: int, callback
    ):
        start = df["date"].min()
        end   = df["date"].max()

        windows = []
        cur = start
        while cur <= end:
            windows.append(cur)
            cur += pd.Timedelta(minutes=slide_time)

        logger.info(f"Number of windows: {len(windows)}")

        all_records = []
        all_metrics = []

        for idx, win_start in enumerate(windows):
            win_end = win_start + pd.Timedelta(minutes=window_time)
            wdf = df[(df["date"] >= win_start) & (df["date"] < win_end)]

            if len(wdf) < config.MIN_EVENTS_PER_WINDOW:
                continue

            try:
                out = self._process_window(wdf.copy(), win_start, idx)
            except Exception as e:
                logger.error(f"Error in window {idx}: {e}")
                continue

            all_records.extend(out["records"])
            all_metrics.append(out["metrics"])

            if callback:
                callback(idx, len(windows), out["metrics"])

        return {
            "anomalies_df": pd.DataFrame(all_records),
            "metrics_df": pd.DataFrame(all_metrics),
        }

    # ------------------------------------------------------------------
    # STEP 3 — PROCESS ONE WINDOW
    # ------------------------------------------------------------------
    def _process_window(self, window_df, win_start, win_idx):
        t0 = time.time()

        # Build texts
        texts = [render_event(r) for _, r in window_df.iterrows()]

        # Embeddings
        embeddings = self.model.get_embeddings(texts, layer=-1, pooling="last")

        # NLL losses
        nll_scores = self.model.compute_nll(texts)
        window_df["llm_nll"] = nll_scores

        # EMA user embeddings
        user_embs = {}
        alpha = 0.3

        for i, row in window_df.iterrows():
            uid = row["user_id"]
            emb = embeddings[i]
            if uid not in user_embs:
                user_embs[uid] = emb.copy()
            else:
                user_embs[uid] = alpha * emb + (1 - alpha) * user_embs[uid]

        zero = np.zeros(self.hidden_size, dtype=np.float32)

        # Build features
        features = []
        for i, row in window_df.iterrows():
            uemb = user_embs.get(row["user_id"], zero)
            pemb = user_embs.get(row["pc_id"], zero)

            extra = np.array([
                row["hour"],
                row["day_of_week"],
                row["activity_encoded"],
                row["role_encoded"],
                row["llm_nll"],
            ], dtype=np.float32)

            features.append(np.concatenate([uemb, pemb, extra]))

        X = np.vstack(features)

        # Rolling baseline anomaly detection
        if len(self.baseline_features) >= config.MIN_BASELINE_SAMPLES:
            baseline = np.vstack(self.baseline_features)
            self.detector.fit(baseline)
            scores, labels = self.detector.predict(X)
        else:
            self.detector.fit(X)
            scores, labels = self.detector.predict(X)

        threshold = self.detector.get_threshold(scores, config.ANOMALY_QUANTILE)

        # Update rolling baseline
        self.baseline_features.extend([x for x in X])
        if len(self.baseline_features) > config.BASELINE_MAX_EVENTS:
            self.baseline_features = self.baseline_features[-config.BASELINE_MAX_EVENTS:]

        # SAFE INDEX LOOP (fix for "index out of bounds")
        n = min(len(window_df), len(scores))
        records = []

        for i in range(n):
            r = window_df.iloc[i]
            records.append({
                "window_index": win_idx,
                "window_start": win_start,
                "date": r["date"],
                "user": r["user"],
                "pc": r["pc"],
                "activity": r["activity"],
                "score": float(scores[i]),
                "is_anomaly": int(scores[i] > threshold),
                "nll": float(r["llm_nll"]),
            })

        n_an = sum(r["is_anomaly"] for r in records)

        metrics = {
            "window_index": win_idx,
            "timestamp": win_start,
            "n_events": len(window_df),
            "n_anomalies": n_an,
            "anomaly_rate": n_an / len(window_df),
            "threshold": float(threshold),
        }

        return {"records": records, "metrics": metrics}

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    def _save_results(self, results: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        anomalies = results["anomalies_df"]
        metrics   = results["metrics_df"]

        anomalies.to_csv(os.path.join(output_dir, "anomalies.csv"), index=False)
        metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

        if not anomalies.empty:
            plt.figure(figsize=(8, 5))
            anomalies["score"].hist(bins=40, edgecolor="black")
            plt.xlabel("Anomaly Score")
            plt.ylabel("Count")
            plt.title("Anomaly Score Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=120)
            plt.close()


# ----------------------------------------------------------------------
# Streamlit wrapper
# ----------------------------------------------------------------------
def run_realtime_detection(
    file_path: str,
    output_base_dir: str,
    test_mode: bool = True,
    window_time: int = 60,
    slide_time: int = 10,
    status_callback=None,
):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(output_base_dir, timestamp)

    pipeline = RealtimePipeline()
    results = pipeline.run(
        input_file=file_path,
        output_dir=output_folder,
        test_mode=test_mode,
        window_time=window_time,
        slide_time=slide_time,
        callback=status_callback,
    )

    return results["anomalies_df"], output_folder


# Manual test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline_rt.py <path_to_csv>")
        sys.exit(0)

    csv_path = sys.argv[1]
    out_dir = os.path.join(config.OUTPUT_BASE_DIR, "manual_test")

    anomalies, out_folder = run_realtime_detection(
        file_path=csv_path,
        output_base_dir=out_dir,
        test_mode=True,
        window_time=60,
        slide_time=10,
        status_callback=None,
    )

    print("Done.")
    print(f"Anomalies: {len(anomalies)}")
    print(f"Outputs in: {out_folder}")

