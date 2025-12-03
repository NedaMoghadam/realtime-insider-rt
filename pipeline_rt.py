"""
Real-Time Detection Pipeline

Connects:
- config.py
- logger.py
- exceptions.py
- validator.py
- data_loader.py
- tinyllama_model.py
- anomaly_detector.py

and exposes:
- class RealtimePipeline
- function run_realtime_detection(...)  ← used by Streamlit app
"""

import os
import time
from datetime import datetime
from typing import Callable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import config
from logger import get_logger
from exceptions import PipelineError
from data_loader import DataLoader
from tinyllama_model import TinyLlamaModel
from anomaly_detector import AnomalyDetector

logger = get_logger(__name__)


def render_event(row: pd.Series) -> str:
    """Turn one log row into a text sentence for TinyLlama."""
    ts = row["date"].strftime("%Y-%m-%d %H:%M")
    user = row.get("user", "unknown")
    role = row.get("role", "unknown")
    pc = row.get("pc", "unknown")
    activity = row.get("activity", "unknown")
    hour = row.get("hour", -1)
    dow = row.get("day_of_week", -1)
    return (
        f"[{ts}] user={user} role={role} pc={pc} "
        f"activity={activity} hour={hour} dow={dow}"
    )


class RealtimePipeline:
    """
    Simple real-time anomaly detection pipeline.

    Steps per time window:
    1) Build event texts
    2) TinyLlama → embeddings (last layer)
    3) Aggregate user/node embeddings
    4) Build edge features [user_emb, pc_emb, activity_encoded]
    5) IsolationForest → anomaly scores + labels
    """

    def __init__(self):
        logger.info("=" * 70)
        logger.info("Initializing Real-Time Pipeline")
        logger.info("=" * 70)

        # 1) Data loader
        self.loader = DataLoader(
            required_columns=config.REQUIRED_COLUMNS,
            optional_columns=config.OPTIONAL_COLUMNS,
            date_column="date",
            date_format=config.DATE_FORMAT,
        )

        # 2) TinyLlama model
        self.model = TinyLlamaModel(
            model_id=config.MODEL_ID,
            cache_dir=None,
            use_gpu=config.USE_GPU,
            max_length=config.MAX_SEQUENCE_LENGTH,
        )

        # 3) Anomaly detector
        self.detector = AnomalyDetector(
            contamination=config.CONTAMINATION_FACTOR
        )

        self.hidden_size = self.model.hidden_size

        logger.info("Pipeline initialization complete")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
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
        """
        Run the pipeline on one CSV file.

        Args:
            input_file: path to CSV
            output_dir: folder to save results
            test_mode: if True, only first N rows (from config.TEST_MODE_ROWS)
            window_time: window size in minutes
            slide_time: slide size in minutes
            callback: optional function(slot_idx, total_slots, info_dict)
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # 1) Load data
        nrows = config.TEST_MODE_ROWS if test_mode else None
        logger.info(f"Loading data from {input_file} (test_mode={test_mode})")
        df = self.loader.load(input_file, nrows=nrows)

        # 2) Preprocess (add features, ids)
        df = self._preprocess(df)

        # 3) Sliding windows
        results = self._process_windows(
            df=df,
            window_time=window_time,
            slide_time=slide_time,
            callback=callback,
        )

        # 4) Save to disk
        self._save_results(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time features, encode categories, and map nodes to ids."""
        logger.info("Preprocessing data...")

        df = df.copy()
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek

        # Label encoders
        activity_enc = LabelEncoder()
        role_enc = LabelEncoder()

        df["activity_encoded"] = activity_enc.fit_transform(df["activity"])
        df["role_encoded"] = role_enc.fit_transform(df["role"])

        # Map users + PCs to integer node ids
        unique_nodes = set(df["user"].tolist() + df["pc"].tolist())
        node_mapping = {node: i for i, node in enumerate(unique_nodes)}
        df["user_id"] = df["user"].map(node_mapping)
        df["pc_id"] = df["pc"].map(node_mapping)

        logger.info(
            f"Preprocessing complete: {len(df)} events, {len(unique_nodes)} unique nodes"
        )
        return df

    def _process_windows(
        self,
        df: pd.DataFrame,
        window_time: int,
        slide_time: int,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> Dict:
        """Main sliding window loop."""
        start_date = df["date"].min()
        end_date = df["date"].max()

        logger.info(
            f"Windowing from {start_date} to {end_date} "
            f"(window={window_time}min, slide={slide_time}min)"
        )

        # Precompute window start times
        window_starts: List[pd.Timestamp] = []
        current = start_date
        while current <= end_date:
            window_starts.append(current)
            current += pd.Timedelta(minutes=slide_time)

        total_windows = len(window_starts)
        logger.info(f"Total windows to process: {total_windows}")

        all_records: List[Dict] = []
        window_metrics: List[Dict] = []

        for idx, win_start in enumerate(window_starts):
            win_end = win_start + pd.Timedelta(minutes=window_time)
            window_df = df[(df["date"] >= win_start) & (df["date"] < win_end)].copy()

            if len(window_df) < config.MIN_EVENTS_PER_WINDOW:
                logger.debug(
                    f"Skipping window {idx} @ {win_start} (only {len(window_df)} events)"
                )
                continue

            try:
                res = self._process_single_window(
                    window_df=window_df,
                    win_start=win_start,
                    win_index=idx,
                )
            except Exception as e:
                logger.error(f"Error in window {idx}: {e}")
                continue

            all_records.extend(res["records"])
            window_metrics.append(res["metrics"])

            # Callback for UI (Streamlit)
            if callback is not None:
                info = {
                    "timestamp": win_start,
                    "events": res["metrics"]["n_events"],
                    "anomalies": res["metrics"]["n_anomalies"],
                    "avg_exit": None,  # we don't use early-exit here yet
                }
                callback(idx, total_windows, info)

        anomalies_df = pd.DataFrame(all_records)
        metrics_df = pd.DataFrame(window_metrics)

        logger.info(f"Processed {total_windows} windows")
        logger.info(f"Total records analyzed: {len(df)}")
        logger.info(f"Total anomalies detected: {len(anomalies_df)}")

        return {
            "anomalies_df": anomalies_df,
            "metrics_df": metrics_df,
            "n_windows": total_windows,
        }

    def _process_single_window(
        self,
        window_df: pd.DataFrame,
        win_start: pd.Timestamp,
        win_index: int,
    ) -> Dict:
        """Process one time window: embeddings → anomaly detection."""
        t0 = time.time()
        window_df = window_df.copy()

        # 1) Build texts
        texts = [render_event(row) for _, row in window_df.iterrows()]

        # 2) TinyLlama embeddings (last token)
        embeddings = self.model.get_embeddings(texts, layer=-1, pooling="last")
        # Optional: NLL (not required for anomaly features)
        nll_scores = self.model.compute_nll(texts)

        # Attach basic stats to rows (as lists, because numpy arrays)
        window_df["embedding"] = list(embeddings)
        window_df["llm_nll"] = nll_scores

        # 3) Aggregate embeddings per user_id
        node_embs: Dict[int, np.ndarray] = {}
        for i, row in window_df.iterrows():
            uid = row["user_id"]
            emb = embeddings[i]
            node_embs.setdefault(uid, []).append(emb)

        for uid, emb_list in node_embs.items():
            node_embs[uid] = np.mean(np.vstack(emb_list), axis=0)

        zero_vec = np.zeros(self.hidden_size, dtype=np.float32)

        # 4) Build edge features: [user_emb, pc_emb, activity_encoded]
        edge_features: List[np.ndarray] = []
        for i, row in window_df.iterrows():
            user_emb = node_embs.get(row["user_id"], zero_vec)
            pc_emb = node_embs.get(row["pc_id"], zero_vec)
            activity_feat = np.array([row["activity_encoded"]], dtype=np.float32)
            feat = np.concatenate([user_emb, pc_emb, activity_feat], axis=0)
            edge_features.append(feat)

        X = np.vstack(edge_features)

        # 5) Anomaly detection
        self.detector.fit(X)
        scores, labels = self.detector.predict(X)
        threshold = self.detector.get_threshold(scores, config.ANOMALY_QUANTILE)

        # 6) Build per-event records
        records: List[Dict] = []
        for i, (_, row) in enumerate(window_df.iterrows()):
            rec = {
                "window_index": win_index,
                "window_start": win_start,
                "date": row["date"],
                "user": row["user"],
                "user_id": row["user_id"],
                "pc": row["pc"],
                "pc_id": row["pc_id"],
                "activity": row["activity"],
                "activity_encoded": row["activity_encoded"],
                "score": float(scores[i]),
                "is_anomaly_if": int(labels[i]),  # IsolationForest label
                "is_anomaly_thr": int(scores[i] > threshold),  # quantile label
                "llm_nll": float(nll_scores[i]),
            }
            records.append(rec)

        n_anomalies = int(sum(r["is_anomaly_thr"] for r in records))
        processing_time = time.time() - t0

        metrics = {
            "window_index": win_index,
            "timestamp": win_start,
            "n_events": len(window_df),
            "n_anomalies": n_anomalies,
            "anomaly_rate": n_anomalies / len(window_df),
            "processing_time_sec": processing_time,
            "threshold": float(threshold),
        }

        logger.debug(
            f"Window {win_index}: {len(window_df)} events, "
            f"{n_anomalies} anomalies, time={processing_time:.2f}s"
        )

        return {"records": records, "metrics": metrics}

    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save anomalies, metrics, and a simple score plot."""
        os.makedirs(output_dir, exist_ok=True)

        anomalies_df: pd.DataFrame = results["anomalies_df"]
        metrics_df: pd.DataFrame = results["metrics_df"]

        anomalies_path = os.path.join(output_dir, "anomalies.csv")
        metrics_path = os.path.join(output_dir, "metrics.csv")

        anomalies_df.to_csv(anomalies_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        logger.info(f"Saved anomalies → {anomalies_path}")
        logger.info(f"Saved metrics   → {metrics_path}")

        # Simple score histogram
        if not anomalies_df.empty:
            plt.figure(figsize=(8, 5))
            anomalies_df["score"].hist(
                bins=config.SCORE_HISTOGRAM_BINS,
                edgecolor="black",
            )
            plt.xlabel("Anomaly score")
            plt.ylabel("Count")
            plt.title("Anomaly Score Distribution")
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "score_distribution.png")
            plt.savefig(plot_path, dpi=config.PLOT_DPI)
            plt.close()

            logger.info(f"Saved score distribution plot → {plot_path}")


# ----------------------------------------------------------------------
# Convenience function used by Streamlit app
# ----------------------------------------------------------------------
def run_realtime_detection(
    file_path: str,
    output_base_dir: str,
    test_mode: bool = True,
    window_time: int = 60,
    slide_time: int = 10,
    status_callback: Optional[Callable[[int, int, Dict], None]] = None,
):
    """
    Wrapper used by the UI.

    Returns:
        anomalies_df, output_folder
    """
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


if __name__ == "__main__":
    # Simple manual test (you can run: python pipeline_rt.py path\to\data.csv)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline_rt.py <path_to_csv>")
        sys.exit(0)

    csv_path = sys.argv[1]
    out_dir = os.path.join(config.OUTPUT_BASE_DIR, "manual_test")

    print(f"Running pipeline on: {csv_path}")
    anomalies, out_folder = run_realtime_detection(
        file_path=csv_path,
        output_base_dir=out_dir,
        test_mode=True,
        window_time=60,
        slide_time=10,
        status_callback=None,
    )
    print(f"Done. Anomalies: {len(anomalies)}")
    print(f"Outputs in: {out_folder}")
