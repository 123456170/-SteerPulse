import os
import math
import time
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB_IMPORT = True
except Exception:
    XGBClassifier = None
    HAS_XGB_IMPORT = False

try:
    import carla
    HAS_CARLA = True
except Exception:
    carla = None
    HAS_CARLA = False

st.set_page_config(page_title="SteerPulse", page_icon="🚘", layout="wide")

SAMPLE_RATE = 20
WINDOW_SECONDS = 30
STRIDE_SECONDS = 5
DEADBAND_DEG = 1.5
REVERSAL_GAP_DEG = 2.5
CORRECTION_THRESHOLD_DEG = 4.0
LANE_WARN_M = 0.45
LOG_DIR = "steerpulse_logs"

FEATURE_COLS = [
    "steering_reversal_rate_per_min",
    "sdlp_m",
    "steering_entropy",
    "zero_crossing_rate_per_s",
    "longest_low_steering_s",
    "correction_magnitude_deg",
    "time_between_corrections_s",
    "mean_speed_mps",
    "std_speed_mps",
    "mean_throttle",
    "max_abs_offset_m",
    "mean_abs_steer_deg",
]


@dataclass
class FatigueEpisode:
    start: float
    end: float
    drift: float
    correction: float
    recover: float
    direction: float


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def driver_step(
    state: Dict[str, float],
    t: float,
    fatigue: Optional[Dict[str, Any]],
    rng: np.random.Generator,
    dt: float = 1.0 / SAMPLE_RATE,
) -> Dict[str, Any]:
    y = float(state["y"])
    y_prev = float(state["y_prev"])
    steer = float(state["steer"])
    speed = float(state["speed"])
    throttle = float(state["throttle"])
    drift_dir = float(state["drift_dir"])

    dy = (y - y_prev) / dt if dt > 0 else 0.0

    wander = (
        1.35 * math.sin(2 * math.pi * t / 8.2)
        + 0.65 * math.sin(2 * math.pi * t / 3.1)
        + 0.35 * math.sin(2 * math.pi * t / 1.7)
    )

    label = "alert"
    event_id = -1

    if fatigue is not None:
        label = "drowsy"
        event_id = int(fatigue.get("event_id", -1))
        ft = t - float(fatigue["start"])
        drift = float(fatigue["drift"])
        corr = float(fatigue["correction"])

        if ft < drift:
            progress = max(0.0, min(1.0, ft / max(drift, 1e-6)))
            drift_rate = 0.032 + 0.042 * progress

            local_dir = drift_dir
            if abs(y) > 0.04:
                local_dir = 1.0 if y > 0 else -1.0

            steer = steer * 0.986 + rng.normal(0.0, 0.10)
            new_y = y + local_dir * drift_rate * dt + rng.normal(0.0, 0.0035)
            throttle = max(0.28, throttle * 0.997 - 0.0008)
            speed = max(23.0, speed - 0.018)

        elif ft < drift + corr:
            sign = 1.0 if y >= 0 else -1.0
            if abs(y) < 0.05:
                sign = drift_dir

            target = -sign * (7.5 + 4.5 * min(abs(y), 1.1))
            steer = steer + (target - steer) * 0.34 + rng.normal(0.0, 0.45)
            new_y = y - sign * (0.28 + 0.08 * rng.random()) * dt + rng.normal(0.0, 0.0035)
            throttle = 0.54 + 0.025 * math.sin(2 * math.pi * t / 4.0) + rng.normal(0.0, 0.012)
            speed = min(34.0, speed + 0.012)

        else:
            target = -1.05 * y + 0.25 * wander
            steer = steer + (target - steer) * 0.24 + rng.normal(0.0, 0.75)
            new_y = y + (-0.42 * y + rng.normal(0.0, 0.030)) * dt
            throttle = 0.52 + 0.025 * math.sin(2 * math.pi * t / 6.0) + rng.normal(0.0, 0.014)
            speed = 30.0 + 1.0 * math.sin(2 * math.pi * t / 15.0) + rng.normal(0.0, 0.12)

    else:
        target = -1.18 * y - 0.22 * dy + wander + rng.normal(0.0, 0.65)
        steer = steer + (target - steer) * 0.48 + rng.normal(0.0, 0.55)
        new_y = y + (-0.52 * y + rng.normal(0.0, 0.034)) * dt + 0.0045 * math.sin(
            2 * math.pi * t / 11.0
        )
        throttle = 0.55 + 0.030 * math.sin(2 * math.pi * t / 9.0) + rng.normal(0.0, 0.014)
        speed = 30.2 + 1.15 * math.sin(2 * math.pi * t / 17.0) + rng.normal(0.0, 0.14)

    steer = _clip(steer, -34.0, 34.0)
    new_y = _clip(new_y, -1.75, 1.75)
    speed = _clip(speed, 18.0, 38.0)
    throttle = _clip(throttle, 0.0, 1.0)

    state["y"] = new_y
    state["y_prev"] = y
    state["steer"] = steer
    state["speed"] = speed
    state["throttle"] = throttle
    state["drift_dir"] = drift_dir

    return {
        "time_s": float(t),
        "steering_angle_deg": float(steer),
        "throttle": float(throttle),
        "speed_mps": float(speed),
        "lateral_offset_m": float(new_y),
        "label": label,
        "event_id": int(event_id),
    }


def make_fatigue_episodes(duration_s: float, rng: np.random.Generator, count: int = 1) -> List[FatigueEpisode]:
    episodes: List[FatigueEpisode] = []
    last_end = 0.0

    for _ in range(int(count)):
        earliest = max(35.0, last_end + 70.0)
        latest = float(duration_s) - 70.0
        if earliest >= latest:
            break

        start = float(rng.uniform(earliest, latest))
        total = float(rng.uniform(30.0, 48.0))
        end = min(float(duration_s) - 5.0, start + total)
        total = end - start

        drift = float(total * rng.uniform(0.50, 0.62))
        corr = float(total * rng.uniform(0.10, 0.16))
        rec = max(2.0, total - drift - corr)
        direction = 1.0 if rng.random() < 0.5 else -1.0

        episodes.append(
            FatigueEpisode(
                start=start,
                end=end,
                drift=drift,
                correction=corr,
                recover=rec,
                direction=direction,
            )
        )
        last_end = end

    return episodes


def simulate_raw_session(
    session_id: int,
    duration_s: float,
    episodes: List[FatigueEpisode],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    state = {
        "y": 0.0,
        "y_prev": 0.0,
        "steer": 0.0,
        "speed": 30.0,
        "throttle": 0.55,
        "drift_dir": 1.0,
    }

    ep_dicts: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for i, ep in enumerate(episodes):
        ep_dicts.append(
            {
                "start": ep.start,
                "end": ep.end,
                "drift": ep.drift,
                "correction": ep.correction,
                "recover": ep.recover,
                "direction": ep.direction,
                "event_id": i,
            }
        )
        event_rows.append(
            {
                "session_id": session_id,
                "event_id": i,
                "start_s": ep.start,
                "end_s": ep.end,
                "drift_s": ep.drift,
                "correction_s": ep.correction,
                "recover_s": ep.recover,
                "direction": ep.direction,
            }
        )

    rows: List[Dict[str, Any]] = []
    n = int(duration_s * SAMPLE_RATE)
    start_ts = pd.Timestamp("2026-01-01 08:00:00") + pd.Timedelta(minutes=session_id * 17)

    for i in range(n):
        t = i / SAMPLE_RATE
        fat = None
        for ed in ep_dicts:
            if ed["start"] <= t < ed["end"]:
                fat = ed
                state["drift_dir"] = ed["direction"]
                break

        sample = driver_step(state, t, fat, rng)
        sample["session_id"] = session_id
        sample["timestamp"] = start_ts + pd.Timedelta(seconds=t)
        rows.append(sample)

    raw = pd.DataFrame(rows)
    events = pd.DataFrame(
        event_rows,
        columns=[
            "session_id",
            "event_id",
            "start_s",
            "end_s",
            "drift_s",
            "correction_s",
            "recover_s",
            "direction",
        ],
    )
    return raw, events


def count_reversals(steer: np.ndarray, gap: float = REVERSAL_GAP_DEG) -> int:
    if len(steer) < 3:
        return 0

    count = 0
    prev_sign = 0
    anchor_idx = 0

    for i in range(1, len(steer)):
        diff = steer[i] - steer[i - 1]
        if abs(diff) < 0.08:
            continue

        sign = 1 if diff > 0 else -1

        if prev_sign == 0:
            prev_sign = sign
            anchor_idx = i - 1
        elif sign != prev_sign:
            if abs(steer[i] - steer[anchor_idx]) > gap:
                count += 1
            prev_sign = sign
            anchor_idx = i - 1

    return count


def zero_crossing_rate(steer: np.ndarray) -> int:
    if len(steer) < 2:
        return 0
    s = np.sign(steer)
    s[s == 0] = 1.0
    return int(np.sum(s[1:] != s[:-1]))


def steering_entropy(steer: np.ndarray, bin_width: float = 2.0) -> float:
    if len(steer) < 5:
        return 0.0

    lo = float(np.min(steer))
    hi = float(np.max(steer))

    if hi - lo < bin_width:
        return 0.0

    start = math.floor(lo / bin_width) * bin_width
    stop = math.ceil(hi / bin_width) * bin_width + bin_width
    bins = np.arange(start, stop + bin_width, bin_width)

    if len(bins) < 2:
        return 0.0

    counts, _ = np.histogram(steer, bins=bins)
    counts = counts[counts > 0].astype(float)

    if counts.size == 0:
        return 0.0

    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def deadband_features(steer: np.ndarray) -> tuple[float, float, float]:
    n = len(steer)
    if n == 0:
        return 0.0, 0.0, float(WINDOW_SECONDS)

    low = np.abs(steer) < DEADBAND_DEG

    longest = 0
    current = 0
    db_times: List[int] = []
    db_mags: List[float] = []

    i = 0
    while i < n:
        if low[i]:
            current += 1
            i += 1
        else:
            if current > 0:
                longest = max(longest, current)
                end_low = i - 1
                seg = steer[i : min(n, i + 40)]
                if len(seg) > 0:
                    mag = float(np.max(np.abs(seg - steer[end_low])))
                    if mag > 2.0:
                        db_mags.append(mag)
                        db_times.append(i)
                current = 0
            i += 1

    if current > 0:
        longest = max(longest, current)

    win = 10
    delta_times: List[int] = []
    delta_mags: List[float] = []

    for i in range(win, n):
        mag = float(np.max(np.abs(steer[i - win : i + 1] - steer[i - win])))
        if mag > CORRECTION_THRESHOLD_DEG:
            if not delta_times or i - delta_times[-1] > 10:
                delta_times.append(i)
                delta_mags.append(mag)

    all_times = sorted(set(db_times + delta_times))
    all_mags = db_mags + delta_mags

    if len(all_times) >= 2:
        diffs = np.diff(all_times) / float(SAMPLE_RATE)
        tbc = float(np.median(diffs))
    else:
        tbc = float(WINDOW_SECONDS)

    if all_mags:
        corr_mag = float(np.mean(all_mags))
    elif n > 1:
        corr_mag = float(np.max(np.abs(np.diff(steer))))
    else:
        corr_mag = 0.0

    return float(longest / SAMPLE_RATE), corr_mag, tbc


def compute_window_features(
    steer: np.ndarray,
    offset: np.ndarray,
    speed: np.ndarray,
    throttle: np.ndarray,
) -> Dict[str, float]:
    dur = len(steer) / float(SAMPLE_RATE)
    if dur <= 0:
        dur = 1e-6

    rev_rate = count_reversals(steer, REVERSAL_GAP_DEG) / max(dur / 60.0, 1e-6)
    sdlp = float(np.std(offset))
    ent = steering_entropy(steer)
    zcr = zero_crossing_rate(steer) / max(dur, 1e-6)
    longest_low, corr_mag, tbc = deadband_features(steer)

    return {
        "steering_reversal_rate_per_min": float(rev_rate),
        "sdlp_m": float(sdlp),
        "steering_entropy": float(ent),
        "zero_crossing_rate_per_s": float(zcr),
        "longest_low_steering_s": float(longest_low),
        "correction_magnitude_deg": float(corr_mag),
        "time_between_corrections_s": float(tbc),
        "mean_speed_mps": float(np.mean(speed)) if len(speed) else 0.0,
        "std_speed_mps": float(np.std(speed)) if len(speed) else 0.0,
        "mean_throttle": float(np.mean(throttle)) if len(throttle) else 0.0,
        "max_abs_offset_m": float(np.max(np.abs(offset))) if len(offset) else 0.0,
        "mean_abs_steer_deg": float(np.mean(np.abs(steer))) if len(steer) else 0.0,
    }


def build_windows(raw: pd.DataFrame) -> pd.DataFrame:
    window_n = int(WINDOW_SECONDS * SAMPLE_RATE)
    stride_n = int(STRIDE_SECONDS * SAMPLE_RATE)
    out: List[Dict[str, Any]] = []

    for sid, g in raw.groupby("session_id", sort=True):
        g = g.sort_values("time_s").reset_index(drop=True)
        steer = g["steering_angle_deg"].to_numpy(dtype=float)
        offset = g["lateral_offset_m"].to_numpy(dtype=float)
        speed = g["speed_mps"].to_numpy(dtype=float)
        throttle = g["throttle"].to_numpy(dtype=float)
        drowsy = (g["label"] == "drowsy").to_numpy(dtype=bool)
        n = len(g)

        if n == 0:
            continue

        if n < window_n:
            starts = [0]
        else:
            starts = list(range(0, n - window_n + 1, stride_n))

        for start in starts:
            end = min(start + window_n, n)
            sl = slice(start, end)
            feats = compute_window_features(steer[sl], offset[sl], speed[sl], throttle[sl])
            frac = float(drowsy[sl].mean())

            out.append(
                {
                    "session_id": sid,
                    "window_start_s": float(start / SAMPLE_RATE),
                    "window_end_s": float(end / SAMPLE_RATE),
                    "label_drowsy": int(frac >= 0.25),
                    "drowsy_fraction": frac,
                    **feats,
                }
            )

    return pd.DataFrame(out)


@st.cache_data(show_spinner="Generating CARLA-compatible highway telemetry...")
def build_demo_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    raw_parts: List[pd.DataFrame] = []
    event_parts: List[pd.DataFrame] = []

    specs = [
        (300, False),
        (300, True),
        (300, False),
        (300, True),
        (300, False),
        (300, True),
    ]

    for sid, (dur, has_fatigue) in enumerate(specs):
        seed = 100 + sid
        rng = np.random.default_rng(seed)
        episodes = make_fatigue_episodes(dur, rng, count=1) if has_fatigue else []
        raw_df, event_df = simulate_raw_session(sid, float(dur), episodes, seed)
        raw_parts.append(raw_df)
        event_parts.append(event_df)

    raw = pd.concat(raw_parts, ignore_index=True)
    events = (
        pd.concat(event_parts, ignore_index=True)
        if event_parts
        else pd.DataFrame(
            columns=[
                "session_id",
                "event_id",
                "start_s",
                "end_s",
                "drift_s",
                "correction_s",
                "recover_s",
                "direction",
            ]
        )
    )

    windows = build_windows(raw)

    log_path = ""
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        raw_path = Path(LOG_DIR) / "demo_raw_telemetry.parquet"
        events_path = Path(LOG_DIR) / "demo_events.parquet"
        windows_path = Path(LOG_DIR) / "demo_windows.parquet"

        raw.to_parquet(raw_path, index=False)
        events.to_parquet(events_path, index=False)
        windows.to_parquet(windows_path, index=False)
        log_path = str(raw_path)
    except Exception:
        log_path = "Parquet logging skipped: check write permissions or pyarrow installation."

    return raw, events, windows, log_path


def compute_detection_delay(
    events: pd.DataFrame,
    windows_eval: pd.DataFrame,
    prob_col: str,
    threshold: float = 0.5,
) -> float:
    if events is None or len(events) == 0 or windows_eval is None or len(windows_eval) == 0:
        return float("nan")

    delays: List[float] = []

    for _, ev in events.iterrows():
        sid = ev["session_id"]
        start = float(ev["start_s"])
        end = float(ev["end_s"])

        cand = windows_eval[
            (windows_eval["session_id"] == sid)
            & (windows_eval[prob_col] >= threshold)
            & (windows_eval["window_end_s"] >= start)
            & (windows_eval["window_start_s"] < end)
        ]

        if len(cand) > 0:
            delays.append(float(cand["window_end_s"].min() - start))

    return float(np.mean(delays)) if delays else float("nan")


@st.cache_resource(show_spinner="Training leave-one-session-out detectors and lane-risk model...")
def train_models(
    raw: pd.DataFrame,
    windows: pd.DataFrame,
    events: pd.DataFrame,
) -> Dict[str, Any]:
    risk_model = None

    try:
        Xr: List[List[float]] = []
        yr: List[float] = []

        for _, g in raw.groupby("session_id"):
            g = g.sort_values("time_s").reset_index(drop=True)
            off = g["lateral_offset_m"].to_numpy(dtype=float)
            ste = g["steering_angle_deg"].to_numpy(dtype=float)
            sp = g["speed_mps"].to_numpy(dtype=float)
            n = len(g)
            lag3 = int(3 * SAMPLE_RATE)
            lag1 = int(1 * SAMPLE_RATE)

            for i in range(lag1, n - lag3):
                Xr.append(
                    [
                        float(off[i]),
                        float(ste[i]),
                        float(sp[i]),
                        float(off[i] - off[i - lag1]),
                        float(ste[i] - ste[i - lag1]),
                    ]
                )
                yr.append(float(off[i + lag3]))

        if Xr:
            risk_model = Ridge(alpha=1.0).fit(np.array(Xr), np.array(yr))
    except Exception:
        risk_model = None

    metrics_template = {
        "Random Forest": {"auc": float("nan"), "delay_s": float("nan"), "n_windows": 0},
        "XGBoost": {"auc": float("nan"), "delay_s": float("nan"), "n_windows": 0},
    }

    if len(windows) == 0:
        return {
            "rf": None,
            "xgb": None,
            "risk": risk_model,
            "metrics": metrics_template,
            "windows_eval": windows,
            "feature_cols": FEATURE_COLS,
        }

    X = windows[FEATURE_COLS].fillna(0).to_numpy(dtype=float)
    y = windows["label_drowsy"].to_numpy(dtype=int)
    groups = windows["session_id"].to_numpy()

    rf_model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )

    xgb_model = None
    if HAS_XGB_IMPORT:
        try:
            xgb_model = XGBClassifier(
                n_estimators=220,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=2,
            )
        except Exception:
            try:
                xgb_model = XGBClassifier()
            except Exception:
                xgb_model = None

    rf_probs = np.full(len(windows), np.nan, dtype=float)
    xgb_probs = np.full(len(windows), np.nan, dtype=float)

    rf_fitted = False
    xgb_fitted = False

    if len(np.unique(y)) > 1:
        for sess in np.unique(groups):
            tr = groups != sess
            te = groups == sess

            if y[tr].sum() == 0 or y[tr].sum() == len(tr):
                continue

            rf_model.fit(X[tr], y[tr])
            rf_probs[te] = rf_model.predict_proba(X[te])[:, 1]
            rf_fitted = True

            if xgb_model is not None:
                xgb_model.fit(X[tr], y[tr])
                xgb_probs[te] = xgb_model.predict_proba(X[te])[:, 1]
                xgb_fitted = True

        if not rf_fitted:
            rf_model.fit(X, y)
            rf_probs = rf_model.predict_proba(X)[:, 1]
            rf_fitted = True

            if xgb_model is not None:
                xgb_model.fit(X, y)
                xgb_probs = xgb_model.predict_proba(X)[:, 1]
                xgb_fitted = True

    if not rf_fitted:
        rf_model = None
    if not xgb_fitted:
        xgb_model = None

    eval_windows = windows.copy()
    eval_windows["rf_prob"] = rf_probs
    eval_windows["xgb_prob"] = xgb_probs

    metrics: Dict[str, Dict[str, Any]] = {}

    for name, col, model in [
        ("Random Forest", "rf_prob", rf_model),
        ("XGBoost", "xgb_prob", xgb_model),
    ]:
        mask = ~np.isnan(eval_windows[col].to_numpy(dtype=float))
        n_eval = int(mask.sum())

        if n_eval > 0 and model is not None:
            yt = y[mask]
            yp = eval_windows.loc[mask, col].to_numpy(dtype=float)
            auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
            delay = compute_detection_delay(events, eval_windows.loc[mask].copy(), col, threshold=0.5)
        else:
            auc = float("nan")
            delay = float("nan")

        metrics[name] = {"auc": auc, "delay_s": delay, "n_windows": n_eval}

    return {
        "rf": rf_model,
        "xgb": xgb_model,
        "risk": risk_model,
        "metrics": metrics,
        "windows_eval": eval_windows,
        "feature_cols": FEATURE_COLS,
    }


def add_event(kind: str, message: str, severity: str = "info") -> None:
    if "event_log" not in st.session_state:
        st.session_state.event_log = []

    st.session_state.event_log.append(
        {
            "clock": pd.Timestamp(datetime.datetime.now()).strftime("%H:%M:%S"),
            "sim_s": round(float(st.session_state.get("sim_time", 0.0)), 1),
            "type": kind,
            "message": message,
            "severity": severity,
        }
    )

    if len(st.session_state.event_log) > 120:
        st.session_state.event_log = st.session_state.event_log[-120:]


def trigger_fatigue(duration: Optional[float] = None, source: str = "manual") -> None:
    if duration is None:
        duration = float(st.session_state.get("fatigue_duration", 32.0))

    duration = float(max(12.0, min(90.0, duration)))
    t = float(st.session_state.get("sim_time", 0.0))

    drift = float(duration * 0.55)
    corr = float(min(4.5, duration * 0.15))
    rec = float(max(2.0, duration - drift - corr))

    y = float(st.session_state.driver_state["y"])
    if abs(y) < 0.03:
        direction = 1.0 if st.session_state.rng.random() < 0.5 else -1.0
    else:
        direction = 1.0 if y > 0 else -1.0

    st.session_state.event_counter = int(st.session_state.get("event_counter", 0)) + 1
    st.session_state.active_fatigue = {
        "start": t,
        "end": t + duration,
        "drift": drift,
        "correction": corr,
        "recover": rec,
        "direction": direction,
        "event_id": st.session_state.event_counter,
    }
    st.session_state.driver_state["drift_dir"] = direction
    st.session_state.auto_fatigue_armed = False

    add_event(
        "Fatigue simulation started",
        f"{source.capitalize()} scripted delay/freeze for {duration:.0f}s "
        f"(drift {drift:.1f}s, correction {corr:.1f}s, recovery {rec:.1f}s).",
        "warning",
    )


def trigger_fatigue_clicked() -> None:
    trigger_fatigue(source="manual")


def reset_live_clicked() -> None:
    keys = [
        "raw_rows",
        "window_rows",
        "event_log",
        "sim_time",
        "active_fatigue",
        "driver_state",
        "rng",
        "last_window_eval",
        "current_risk",
        "event_counter",
        "auto_fatigue_armed",
        "live_initialized",
        "model_name",
        "threshold",
        "fatigue_trigger_at",
        "fatigue_duration",
        "auto_fatigue",
        "running",
    ]
    for k in keys:
        st.session_state.pop(k, None)


def save_live_logs() -> Optional[str]:
    rows = st.session_state.get("raw_rows", [])
    if not rows:
        return None

    df = pd.DataFrame(rows)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(LOG_DIR) / f"live_session_{stamp}.parquet"
    df.to_parquet(path, index=False)

    wins = st.session_state.get("window_rows", [])
    if wins:
        pd.DataFrame(wins).to_parquet(path.with_name(path.stem + "_windows.parquet"), index=False)

    evs = st.session_state.get("event_log", [])
    if evs:
        pd.DataFrame(evs).to_parquet(path.with_name(path.stem + "_events.parquet"), index=False)

    return str(path)


def save_live_logs_clicked() -> None:
    path = save_live_logs()
    st.session_state.saved_live_path = path if path else "No live rows to save."


def advance_one_live_sample() -> Dict[str, Any]:
    t = float(st.session_state.sim_time)
    state = st.session_state.driver_state
    rng = st.session_state.rng
    fat = st.session_state.active_fatigue

    sample = driver_step(state, t, fat, rng)
    sample["session_id"] = "live"
    sample["timestamp"] = pd.Timestamp(datetime.datetime.now()) + pd.Timedelta(seconds=t)

    st.session_state.raw_rows.append(sample)
    st.session_state.sim_time = t + 1.0 / SAMPLE_RATE

    max_rows = int(900 * SAMPLE_RATE)
    if len(st.session_state.raw_rows) > max_rows:
        st.session_state.raw_rows = st.session_state.raw_rows[-max_rows:]

    return sample


def maybe_manage_fatigue() -> None:
    t = float(st.session_state.sim_time)
    af = st.session_state.active_fatigue

    if af is not None and t >= float(af["end"]):
        st.session_state.active_fatigue = None
        add_event(
            "Fatigue episode ended",
            "Scripted driver-input delay/freeze/correction episode completed.",
            "info",
        )

    if (
        st.session_state.get("auto_fatigue", True)
        and st.session_state.get("auto_fatigue_armed", True)
        and t >= float(st.session_state.get("fatigue_trigger_at", 35.0))
    ):
        trigger_fatigue(float(st.session_state.get("fatigue_duration", 32.0)), source="auto")


def compute_live_risk(
    recent_df: pd.DataFrame,
    feats: Dict[str, Any],
    risk_model: Optional[Any],
) -> Dict[str, Any]:
    last = recent_df.iloc[-1]
    cur = float(last["lateral_offset_m"])

    if risk_model is None or len(recent_df) < 2:
        pred = cur
    else:
        prev_idx = max(0, len(recent_df) - 1 - int(1 * SAMPLE_RATE))
        prev = recent_df.iloc[prev_idx]
        X = np.array(
            [
                [
                    cur,
                    float(last["steering_angle_deg"]),
                    float(last["speed_mps"]),
                    cur - float(prev["lateral_offset_m"]),
                    float(last["steering_angle_deg"]) - float(prev["steering_angle_deg"]),
                ]
            ]
        )
        pred = float(risk_model.predict(X)[0])

    future_abs = max(abs(cur), abs(pred))
    rising = abs(pred) > abs(cur) + 0.04

    z = max(-50.0, min(50.0, (future_abs - 0.36) * 11.0))
    base = 1.0 / (1.0 + math.exp(-z))

    sdlp = float(feats.get("sdlp_m", 0.0))
    low = float(feats.get("longest_low_steering_s", 0.0))
    rev = float(feats.get("steering_reversal_rate_per_min", 20.0))

    sdlp_bonus = min(0.28, sdlp * 0.75)
    low_bonus = min(0.22, (low / 30.0) * 0.22)
    rev_bonus = min(0.18, max(0.0, 8.0 - rev) / 8.0 * 0.18)

    score = 100.0 * max(0.0, min(1.0, base + sdlp_bonus + low_bonus + rev_bonus))
    flag = score >= 65.0 or (rising and future_abs >= 0.45)

    return {
        "score": float(score),
        "predicted_offset_m": float(pred),
        "current_offset_m": float(cur),
        "rising": bool(rising),
        "flag": bool(flag),
    }


def maybe_evaluate_live_window() -> None:
    t = float(st.session_state.sim_time)
    if t < WINDOW_SECONDS:
        return

    candidate = math.floor(t / STRIDE_SECONDS) * STRIDE_SECONDS
    if candidate <= float(st.session_state.last_window_eval) + 1e-6:
        return

    need = int(WINDOW_SECONDS * SAMPLE_RATE)
    rows = st.session_state.raw_rows[-need:]
    if len(rows) < need:
        return

    df = pd.DataFrame(rows)
    feats = compute_window_features(
        df["steering_angle_deg"].to_numpy(dtype=float),
        df["lateral_offset_m"].to_numpy(dtype=float),
        df["speed_mps"].to_numpy(dtype=float),
        df["throttle"].to_numpy(dtype=float),
    )

    bundle = st.session_state.bundle
    X = pd.DataFrame([feats], columns=bundle["feature_cols"]).fillna(0)

    model_key = "rf" if str(st.session_state.model_name).startswith("Random") else "xgb"
    model = bundle.get(model_key)
    prob = float("nan")

    if model is not None:
        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception:
            prob = float("nan")

    risk = compute_live_risk(df, feats, bundle.get("risk"))

    row = {
        "sim_time_s": float(t),
        "window_end_s": float(t),
        **feats,
        "probability": prob,
        "risk_score": risk["score"],
        "risk_flag": risk["flag"],
        "detected": int(prob >= float(st.session_state.threshold)) if not math.isnan(prob) else 0,
    }

    st.session_state.window_rows.append(row)
    st.session_state.last_window_eval = float(candidate)

    if len(st.session_state.window_rows) > 240:
        st.session_state.window_rows = st.session_state.window_rows[-240:]

    if row["detected"]:
        add_event(
            "Drowsy detection",
            f"{st.session_state.model_name} probability {prob:.2f} exceeded threshold "
            f"{float(st.session_state.threshold):.2f}.",
            "error",
        )

    if row["risk_flag"]:
        add_event(
            "Lane departure risk",
            f"Predicted 3s lateral offset {risk['predicted_offset_m']:+.2f} m; "
            f"risk {risk['score']:.0f}/100.",
            "warning",
        )


def update_current_risk() -> None:
    rows = st.session_state.raw_rows[-int(WINDOW_SECONDS * SAMPLE_RATE) :]
    if len(rows) < 20:
        latest_offset = float(rows[-1]["lateral_offset_m"]) if rows else 0.0
        st.session_state.current_risk = {
            "score": 0.0,
            "predicted_offset_m": latest_offset,
            "current_offset_m": latest_offset,
            "rising": False,
            "flag": False,
        }
        return

    df = pd.DataFrame(rows)

    if st.session_state.window_rows:
        feats = st.session_state.window_rows[-1]
    else:
        feats = compute_window_features(
            df["steering_angle_deg"].to_numpy(dtype=float),
            df["lateral_offset_m"].to_numpy(dtype=float),
            df["speed_mps"].to_numpy(dtype=float),
            df["throttle"].to_numpy(dtype=float),
        )

    st.session_state.current_risk = compute_live_risk(df, feats, st.session_state.bundle.get("risk"))


def advance_live_batch(step: float = 0.5) -> None:
    if not st.session_state.get("running", False):
        return

    n = int(round(step * SAMPLE_RATE))
    for _ in range(max(1, n)):
        maybe_manage_fatigue()
        advance_one_live_sample()

    maybe_evaluate_live_window()
    update_current_risk()


def init_state(bundle: Dict[str, Any], demo_log_path: str) -> None:
    defaults = {
        "running": True,
        "auto_fatigue": True,
        "auto_fatigue_armed": True,
        "fatigue_trigger_at": 35.0,
        "fatigue_duration": 32.0,
        "model_name": "Random Forest",
        "threshold": 0.50,
        "event_counter": 0,
        "saved_live_path": "",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.session_state["bundle"] = bundle
    st.session_state["demo_log_path"] = demo_log_path

    if st.session_state.get("live_initialized", False):
        return

    st.session_state.raw_rows = []
    st.session_state.window_rows = []
    st.session_state.event_log = []
    st.session_state.sim_time = 0.0
    st.session_state.active_fatigue = None
    st.session_state.last_window_eval = -999.0
    st.session_state.current_risk = {
        "score": 0.0,
        "predicted_offset_m": 0.0,
        "current_offset_m": 0.0,
        "rising": False,
        "flag": False,
    }
    st.session_state.driver_state = {
        "y": 0.0,
        "y_prev": 0.0,
        "steer": 0.0,
        "speed": 30.2,
        "throttle": 0.55,
        "drift_dir": 1.0,
    }
    st.session_state.rng = np.random.default_rng(2026)

    for _ in range(int(32 * SAMPLE_RATE)):
        advance_one_live_sample()

    st.session_state.live_initialized = True

    maybe_evaluate_live_window()
    update_current_risk()
    add_event(
        "System",
        "SteerPulse live demo initialized with 32 seconds of normal highway telemetry.",
        "info",
    )


def make_lane_view(df: pd.DataFrame) -> go.Figure:
    latest = df.iloc[-1]
    trail = df.tail(120)

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=-1.8,
        x1=1.8,
        y0=-1.45,
        y1=0.45,
        fillcolor="rgb(38,39,43)",
        line_color="rgb(90,90,90)",
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=[-1.8, -1.8],
            y=[-1.45, 0.45],
            mode="lines",
            line=dict(color="white", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1.8, 1.8],
            y=[-1.45, 0.45],
            mode="lines",
            line=dict(color="white", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-1.45, 0.45],
            mode="lines",
            line=dict(color="yellow", dash="dot", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if len(trail) > 1:
        fig.add_trace(
            go.Scatter(
                x=trail["lateral_offset_m"],
                y=np.linspace(-1.15, 0.0, len(trail)),
                mode="lines",
                line=dict(color="rgba(0,209,255,0.35)", width=10),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[float(latest["lateral_offset_m"])],
            y=[0.0],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=24,
                color="crimson",
                line=dict(color="white", width=1),
                angle=-float(latest["steering_angle_deg"]) * 0.35,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="CARLA-compatible highway view", font=dict(size=14)),
        xaxis=dict(range=[-2.25, 2.25], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.55, 0.55], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=8, r=8, t=38, b=8),
        height=270,
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        uirevision="lane-view",
    )

    return fig


def make_offset_trace(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(
        y0=-LANE_WARN_M,
        y1=LANE_WARN_M,
        fillcolor="rgba(0,220,0,0.08)",
        line_width=0,
        layer="below",
    )
    fig.add_hline(y=LANE_WARN_M, line_dash="dash", line_color="orange", line_width=1)
    fig.add_hline(y=-LANE_WARN_M, line_dash="dash", line_color="orange", line_width=1)

    fig.add_trace(
        go.Scatter(
            x=df["rel_t"],
            y=df["lateral_offset_m"],
            mode="lines",
            name="Lateral offset",
            line=dict(color="#00d1ff", width=2),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Lane offset trace, last 60 seconds", font=dict(size=14)),
        xaxis_title="seconds ago",
        yaxis_title="meters",
        height=245,
        margin=dict(l=35, r=10, t=38, b=25),
        showlegend=False,
        uirevision="offset",
    )

    return fig


def make_steering_trace(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(
        y0=-DEADBAND_DEG,
        y1=DEADBAND_DEG,
        fillcolor="rgba(255,255,0,0.08)",
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=df["rel_t"],
            y=df["steering_angle_deg"],
            mode="lines",
            name="Steering angle",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Steering angle trace, last 60 seconds", font=dict(size=14)),
        xaxis_title="seconds ago",
        yaxis_title="degrees",
        height=245,
        margin=dict(l=35, r=10, t=38, b=25),
        showlegend=False,
        uirevision="steer",
    )

    return fig


def make_risk_gauge(score: float) -> go.Figure:
    try:
        value = float(score)
        if math.isnan(value):
            value = 0.0
    except Exception:
        value = 0.0

    value = max(0.0, min(100.0, value))

    if value >= 65:
        bar_color = "crimson"
    elif value >= 35:
        bar_color = "orange"
    else:
        bar_color = "limegreen"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "/100", "font": {"size": 30}},
            title={"text": "Lane-departure / drowsiness risk"},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 35, 65, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 35], "color": "rgba(0,200,0,0.18)"},
                    {"range": [35, 65], "color": "rgba(255,165,0,0.20)"},
                    {"range": [65, 100], "color": "rgba(220,20,60,0.24)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "value": 65,
                },
            },
        )
    )

    fig.update_layout(
        height=245,
        margin=dict(l=20, r=20, t=45, b=10),
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
    )

    return fig


def render_live_dashboard() -> None:
    st.subheader("Live telemetry dashboard")

    if "raw_rows" not in st.session_state or not st.session_state.raw_rows:
        st.info("Warming up live demo...")
        return

    recent_rows = st.session_state.raw_rows[-int(90 * SAMPLE_RATE) :]
    df = pd.DataFrame(recent_rows)
    latest_time = float(st.session_state.raw_rows[-1]["time_s"])

    plot_df = df[df["time_s"] >= latest_time - 60].copy()
    if plot_df.empty:
        plot_df = df.copy()
    plot_df["rel_t"] = plot_df["time_s"] - latest_time

    window_df = (
        pd.DataFrame(st.session_state.window_rows[-60:])
        if st.session_state.window_rows
        else pd.DataFrame()
    )

    latest_window = window_df.iloc[-1] if len(window_df) else None
    risk = st.session_state.current_risk
    threshold = float(st.session_state.threshold)

    prob = float("nan")
    if latest_window is not None and "probability" in latest_window and not pd.isna(latest_window["probability"]):
        prob = float(latest_window["probability"])

    detected = (not math.isnan(prob)) and prob >= threshold

    if detected:
        status_text = "DROWSY DETECTED"
    elif st.session_state.active_fatigue is not None:
        status_text = "FATIGUE SCRIPT ACTIVE"
    else:
        status_text = "ALERT"

    col_view, col_gauge, col_metrics = st.columns([1.05, 0.90, 2.05])

    with col_view:
        st.plotly_chart(
            make_lane_view(plot_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_gauge:
        st.plotly_chart(
            make_risk_gauge(float(risk.get("score", 0.0))),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            f"Predicted 3s lateral offset: {float(risk.get('predicted_offset_m', 0.0)):+.2f} m | "
            f"Early flag: {'YES' if risk.get('flag', False) else 'no'}"
        )

    with col_metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sim time", f"{latest_time:.1f}s")
        c2.metric("Status", status_text)
        c3.metric("Model prob", "n/a" if math.isnan(prob) else f"{prob:.2f}")

        if latest_window is not None:
            c4.metric(
                "Reversal rate",
                f"{float(latest_window['steering_reversal_rate_per_min']):.1f}/min",
            )
            c5.metric("SDLP", f"{float(latest_window['sdlp_m']):.2f} m")
        else:
            c4.metric("Reversal rate", "waiting")
            c5.metric("SDLP", "waiting")

    col_off, col_steer = st.columns(2)

    with col_off:
        st.plotly_chart(
            make_offset_trace(plot_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_steer:
        st.plotly_chart(
            make_steering_trace(plot_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    col_feat, col_events = st.columns([1.35, 1.0])

    with col_feat:
        if len(window_df) > 1:
            feat_plot = window_df.copy()
            feat_plot["rel_t"] = feat_plot["sim_time_s"] - latest_time

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=feat_plot["rel_t"],
                    y=feat_plot["steering_reversal_rate_per_min"],
                    mode="lines+markers",
                    name="Reversal rate / min",
                    line=dict(color="#2ca02c", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=feat_plot["rel_t"],
                    y=feat_plot["sdlp_m"] * 20.0,
                    mode="lines+markers",
                    name="SDLP m x20",
                    line=dict(color="#d62728", width=2, dash="dot"),
                )
            )
            fig.update_layout(
                template="plotly_dark",
                title=dict(text="30-second window features", font=dict(size=14)),
                xaxis_title="seconds ago",
                height=250,
                margin=dict(l=35, r=10, t=38, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                uirevision="features",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Waiting for enough 30-second sliding windows...")

    with col_events:
        st.markdown("**Event list**")
        if st.session_state.event_log:
            ev = pd.DataFrame(st.session_state.event_log).iloc[::-1]
            st.dataframe(ev, use_container_width=True, height=250, hide_index=True)
        else:
            st.caption("No events yet.")


def live_loop() -> None:
    if st.session_state.get("running", False):
        advance_live_batch(0.5)
    render_live_dashboard()


def render_sidebar(bundle: Dict[str, Any]) -> None:
    st.sidebar.title("SteerPulse controls")

    st.sidebar.markdown(
        f"**CARLA Python API:** {'detected' if HAS_CARLA else 'not detected; using synthetic CARLA-compatible source'}"
    )

    if HAS_CARLA:
        if st.sidebar.button("Test CARLA server"):
            try:
                client = carla.Client("localhost", 2000)
                client.set_timeout(2.0)
                world = client.get_world()
                st.sidebar.success(f"CARLA reachable: {world.get_map().name}")
            except Exception as exc:
                st.sidebar.error(f"CARLA connection failed: {exc}")

    st.sidebar.checkbox("Running", key="running")
    st.sidebar.button("Trigger fatigue now", on_click=trigger_fatigue_clicked)

    st.sidebar.checkbox("Auto demo fatigue", key="auto_fatigue")
    st.sidebar.number_input(
        "Auto fatigue at sim second",
        min_value=5.0,
        max_value=300.0,
        step=1.0,
        key="fatigue_trigger_at",
        format="%.1f",
    )
    st.sidebar.slider(
        "Fatigue duration seconds",
        min_value=15,
        max_value=75,
        step=1,
        key="fatigue_duration",
    )

    options = ["Random Forest"]
    if bundle.get("xgb") is not None:
        options.append("XGBoost")

    if st.session_state.get("model_name") not in options:
        st.session_state.model_name = options[0]

    st.sidebar.selectbox("Detector model", options, key="model_name")
    st.sidebar.slider(
        "Detection threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="threshold",
    )

    st.sidebar.button("Reset live demo", on_click=reset_live_clicked)
    st.sidebar.button("Save live Parquet log", on_click=save_live_logs_clicked)

    if st.session_state.get("saved_live_path"):
        st.sidebar.success(str(st.session_state.saved_live_path))

    st.sidebar.caption(f"Demo raw log: `{st.session_state.get('demo_log_path', '')}`")


def render_model_diagnostics(bundle: Dict[str, Any]) -> None:
    with st.expander("Model validation: leave-one-session-out ROC-AUC and detection delay", expanded=False):
        rows = []
        for name, met in bundle["metrics"].items():
            rows.append(
                {
                    "Model": name,
                    "ROC-AUC": met["auc"],
                    "Mean detection delay s": met["delay_s"],
                    "Eval windows": met["n_windows"],
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Detection delay is measured from scripted drowsy onset to the end of the first "
            "30-second sliding window flagged positive at threshold 0.50."
        )

        model_key = "rf" if str(st.session_state.model_name).startswith("Random") else "xgb"
        model = bundle.get(model_key)

        if model is not None and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=bundle["feature_cols"]).sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
            fig.update_layout(
                template="plotly_dark",
                title=dict(text=f"Feature importance: {st.session_state.model_name}", font=dict(size=14)),
                height=330,
                margin=dict(l=170, r=15, t=40, b=25),
                xaxis_title="normalized importance",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_limitations() -> None:
    with st.expander("Limitations and real-road validation", expanded=False):
        st.markdown(
            """
            - **Simulator-trained models need real-road validation.** This demo uses a CARLA-compatible
              synthetic highway source so the app runs immediately without a CARLA server. Models trained
              here should not be deployed on real vehicles without prospective on-road evaluation.
            - **Scripted fatigue is not physiological fatigue.** The demo labels windows from scripted
              input delay/freeze/correction patterns. Real drowsiness involves gaze, head pose, eyelid
              closure, reaction time, circadian state, medication, workload, and individual differences.
            - **Telemetry-only detection is confounded.** Road curvature, surface friction, crosswinds,
              tire pressure, cruise-control behavior, ADAS interventions, and driver style can all change
              steering and lane-position statistics.
            - **Lane offset depends on calibration.** Camera/lane-detection calibration, GPS/INS accuracy,
              map matching, and coordinate conventions materially affect SDLP and departure-risk estimates.
            - **Detection delay has a hard floor from windowing.** A 30-second sliding window cannot provide
              sub-window latency. The lane-risk estimator gives earlier analog warning, but it is still a
              proxy, not a certified safety system.
            - **Privacy and governance.** Vehicle telemetry can reveal location, driving style, and health-
              related inferences. Any real deployment needs consent, data minimization, retention limits,
              and regulatory review.
            """
        )


def main() -> None:
    st.title("🚘 SteerPulse")
    st.caption(
        "Telemetry-only drowsiness detection using steering dynamics, lane-position variability, "
        "dead-band/correction features, and an early lane-departure risk estimator."
    )

    raw, events, windows_demo, demo_log_path = build_demo_dataset()
    bundle = train_models(raw, windows_demo, events)

    init_state(bundle, demo_log_path)
    render_sidebar(bundle)

    if hasattr(st, "fragment"):
        live_fragment = st.fragment(run_every=0.5)(live_loop)
        live_fragment()
    else:
        live_loop()

    render_model_diagnostics(bundle)
    render_limitations()

    if not hasattr(st, "fragment") and st.session_state.get("running", False):
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()