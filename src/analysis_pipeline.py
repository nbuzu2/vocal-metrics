from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from typing import Callable

import librosa
import numpy as np
import pandas as pd

from app_io import load_audio
from metrics import analyze_pitch_range
from metrics import analyze_vibrato
from metrics import extract_frame_features


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    numeric = float(value)
    if not np.isfinite(numeric):
        return default
    return numeric


def _safe_note_from_hz(value: Any) -> str | None:
    hz = _safe_float(value, default=0.0)
    if hz <= 0:
        return None
    return librosa.hz_to_note(hz, octave=True, cents=False)


def _progress_print(
    message: str,
    enabled: bool = False,
    collector: Callable[[str], None] | None = None,
) -> None:
    if collector is not None:
        collector(message)
    if enabled:
        print(message, file=sys.stderr, flush=True)

# TODO: classify methods v alpha, next use more accurate
def _classify_vibrato(rate_hz: float, extent_hz: float) -> str:
    if rate_hz == 0:
        return "insuficiente para analizar"
    if rate_hz < 3:
        return "sin vibrato o muy lento"
    if 4 <= rate_hz <= 7:
        return "natural/controlado" if extent_hz < 15 else "amplio"
    if rate_hz > 7:
        return "rapido/tembloroso"
    return "irregular"


def _classify_breath_control(rms_cv: float) -> str:
    if rms_cv < 0.3:
        return "muy estable"
    if rms_cv <= 0.7:
        return "buen control dinamico"
    return "muy variable"


def _classify_brightness(mean_centroid: float) -> str:
    if mean_centroid < 2000:
        return "oscuro/calido"
    if mean_centroid <= 3500:
        return "equilibrado"
    return "brillante/claro"


def _classify_tone_quality(mean_flatness: float) -> str:
    if mean_flatness < 0.001:
        return "tono puro"
    if mean_flatness <= 0.003:
        return "buen tono"
    return "tono ruidoso"


def _build_per_second(df: pd.DataFrame) -> list[dict[str, Any]]:
    df_seconds = df[np.isfinite(df["time_s"])].copy()
    df_seconds["second"] = df_seconds["time_s"].astype(int)

    rows: list[dict[str, Any]] = []
    for second, chunk in df_seconds.groupby("second", sort=True):
        voiced = chunk[chunk["is_voice"]]
        voiced_freq = voiced["frequency_smoothed"].dropna()
        voiced_freq = voiced_freq[np.isfinite(voiced_freq) & (voiced_freq > 0)]
        mean_frequency = voiced_freq.mean() if not voiced_freq.empty else np.nan
        mean_note = _safe_note_from_hz(mean_frequency)

        rows.append(
            {
                "second": int(second),
                "time_start_s": float(second),
                "time_end_s": float(second + 1),
                "voice_percentage": round(100 * chunk["is_voice"].mean(), 1),
                "mean_voiced_prob": round(_safe_float(chunk["voiced_prob"].mean()), 4),
                "mean_rms": round(_safe_float(chunk["rms"].mean()), 6),
                "mean_pitch_hz": round(_safe_float(mean_frequency), 2) if np.isfinite(mean_frequency) else None,
                "mean_note": mean_note,
                "mean_centroid_hz": round(_safe_float(chunk["centroid_hz"].mean()), 2),
                "mean_flatness": round(_safe_float(chunk["flatness"].mean()), 6),
            }
        )

    return rows


def _build_frame_by_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    voiced_df = df[df["is_voice"]].copy()
    for row in voiced_df.itertuples(index=False):
        frame_rows.append(
            {
                "time_s": round(_safe_float(row.time_s), 4),
                "is_voice": bool(row.is_voice),
                "voiced_prob": round(_safe_float(row.voiced_prob), 4),
                "rms": round(_safe_float(row.rms), 6),
                "frequency_hz": round(_safe_float(row.frequency_hz), 2) if np.isfinite(_safe_float(row.frequency_hz, np.nan)) else None,
                "frequency_smoothed": round(_safe_float(row.frequency_smoothed), 2) if np.isfinite(_safe_float(row.frequency_smoothed, np.nan)) and _safe_float(row.frequency_smoothed) > 0 else None,
                "note_name": row.note_name if isinstance(row.note_name, str) else None,
                "centroid_hz": round(_safe_float(row.centroid_hz), 2),
                "flatness": round(_safe_float(row.flatness), 6),
            }
        )
    return frame_rows


def _build_summary(df: pd.DataFrame, sr: int, duration: float) -> dict[str, Any]:
    valid_pitch = df[df["is_voice"]]["frequency_smoothed"].dropna()
    valid_pitch = valid_pitch[np.isfinite(valid_pitch) & (valid_pitch > 0)]
    pitch_stats = analyze_pitch_range(df)
    vibrato_stats = analyze_vibrato(df, sr)

    mean_hz = _safe_float(valid_pitch.mean()) if not valid_pitch.empty else 0.0
    std_hz = _safe_float(valid_pitch.std()) if not valid_pitch.empty else 0.0
    mean_note = _safe_note_from_hz(mean_hz)
    p5_hz = _safe_float(valid_pitch.quantile(0.05)) if not valid_pitch.empty else 0.0
    p95_hz = _safe_float(valid_pitch.quantile(0.95)) if not valid_pitch.empty else 0.0
    p5_note = _safe_note_from_hz(p5_hz)
    p95_note = _safe_note_from_hz(p95_hz)

    voiced_df = df[df["is_voice"]]
    rms_mean = _safe_float(voiced_df["rms"].mean())
    rms_std = _safe_float(voiced_df["rms"].std())
    rms_cv = rms_std / rms_mean if rms_mean > 0 else 0.0
    mean_centroid = _safe_float(voiced_df["centroid_hz"].mean())
    mean_flatness = _safe_float(voiced_df["flatness"].mean())
    vibrato_rate = _safe_float(vibrato_stats["vibrato_rate_hz"])
    vibrato_extent = _safe_float(vibrato_stats["vibrato_extent_hz"])

    return {
        "duration": round(duration, 3),
        "sample_rate": int(sr),
        "frame_count": int(len(df)),
        "voice_percentage": round(100 * df["is_voice"].mean(), 1) if len(df) else 0.0,
        "min_note": pitch_stats["min_note"],
        "max_note": pitch_stats["max_note"],
        "mean_note": mean_note,
        "range_semitones": round(_safe_float(pitch_stats["range_semitones"]), 2),
        "range_robust_semitones": round(_safe_float(pitch_stats["range_robust_semitones"]), 2),
        "note_p5": p5_note,
        "note_p95": p95_note,
        "pitch_mean_hz": round(mean_hz, 2),
        "pitch_std_hz": round(std_hz, 2),
        "vibrato_rate_hz": round(vibrato_rate, 2),
        "vibrato_extent_hz": round(vibrato_extent, 2),
        "vibrato_class": _classify_vibrato(vibrato_rate, vibrato_extent),
        "rms_cv": round(rms_cv, 3),
        "breath_control": _classify_breath_control(rms_cv),
        "mean_centroid_hz": round(mean_centroid, 1),
        "brightness": _classify_brightness(mean_centroid),
        "mean_flatness": round(mean_flatness, 4),
        "tone_quality": _classify_tone_quality(mean_flatness),
    }


def _build_report_lines(summary: dict[str, Any]) -> list[str]:
    return [
        "REPORTE DE ANALISIS VOCAL",
        f"Duracion: {summary['duration']:.1f}s ({summary['voice_percentage']}% con voz)",
        f"Rango vocal: {summary['min_note']} a {summary['max_note']} | media {summary['mean_note']}",
        f"Rango total: {summary['range_semitones']} semitonos | tipico {summary['note_p5']} a {summary['note_p95']}",
        f"Vibrato: {summary['vibrato_rate_hz']} Hz | amplitud {summary['vibrato_extent_hz']} Hz | {summary['vibrato_class']}",
        f"Respiracion: CV RMS {summary['rms_cv']} | {summary['breath_control']}",
        f"Timbre: centroide {summary['mean_centroid_hz']} Hz ({summary['brightness']}) | flatness {summary['mean_flatness']} ({summary['tone_quality']})",
    ]


def _build_progress_lines(summary: dict[str, Any]) -> list[str]:
    return [
        "🎼 RANGO VOCAL:",
        f"   Nota mas baja: {summary['min_note']}",
        f"   Nota mas alta: {summary['max_note']}",
        f"   Nota promedio: {summary['mean_note']}",
        f"   Rango total: {summary['range_semitones']} semitonos",
        f"   Rango tipico (5-95%): {summary['note_p5']} a {summary['note_p95']} ({summary['range_robust_semitones']} semitonos)",
        "",
        "🌊 VIBRATO:",
        f"   Frecuencia: {summary['vibrato_rate_hz']:.2f} Hz (ciclos/seg)",
        f"   Amplitud: {summary['vibrato_extent_hz']:.2f} Hz",
        f"   Clasificacion: {summary['vibrato_class']}",
        "",
        "💨 CONTROL DE RESPIRACION:",
        f"   Variabilidad: {summary['rms_cv']:.3f}",
        f"   Evaluacion: {summary['breath_control']}",
        "",
        "✨ TIMBRE:",
        f"   Brillo: {summary['mean_centroid_hz']:.0f} Hz -> {summary['brightness']}",
        f"   Pureza: {summary['mean_flatness']:.4f} -> {summary['tone_quality']}",
        "",
        "✅ Analisis completado",
    ]


def analyze_audio(y: np.ndarray, sr: int, hop_length: int = 512, include_frame_details: bool = False) -> dict[str, Any]:
    """Run the full metrics pipeline from in-memory audio."""
    return analyze_audio_with_progress(
        y,
        sr,
        hop_length=hop_length,
        include_frame_details=include_frame_details,
    )


def analyze_audio_with_progress(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    include_frame_details: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full metrics pipeline from in-memory audio with optional progress callbacks."""
    progress_enabled = include_frame_details
    progress_messages: list[str] = []
    duration = float(len(y) / sr) if sr else 0.0
    _progress_print(
        "Iniciando extraccion de features por frame...",
        progress_enabled,
        lambda message: _collect_progress(progress_messages, progress_callback, message),
    )
    df = extract_frame_features(y, sr, hop_length=hop_length)
    invalid_time_frames = int((~np.isfinite(df["time_s"])).sum()) if "time_s" in df else 0
    if invalid_time_frames:
        _progress_print(
            f"Se detectaron {invalid_time_frames} frames con tiempo no finito; se excluiran del resumen por segundo.",
            progress_enabled,
            lambda message: _collect_progress(progress_messages, progress_callback, message),
        )
    _progress_print(
        "Construyendo resumen general...",
        progress_enabled,
        lambda message: _collect_progress(progress_messages, progress_callback, message),
    )
    summary = _build_summary(df, sr, duration)
    for line in _build_progress_lines(summary):
        _progress_print(
            line,
            progress_enabled,
            lambda message: _collect_progress(progress_messages, progress_callback, message),
        )
    result = {
        "summary": summary,
        "report_lines": _build_report_lines(summary),
        "progress_messages": progress_messages,
        "analysis_mode": {
            "hop_length": hop_length,
            "include_frame_details": include_frame_details,
        },
    }
    if include_frame_details:
        _progress_print(
            "Generando detalle frame a frame...",
            progress_enabled,
            lambda message: _collect_progress(progress_messages, progress_callback, message),
        )
        result["frame_by_frame"] = _build_frame_by_frame(df)
    else:
        result["second_by_second"] = _build_per_second(df)
    return result


def analyze_audio_file(
    path: str | Path,
    hop_length: int = 512,
    include_frame_details: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load an audio file from disk and analyze it."""
    resolved_path = Path(path).expanduser()
    y, sr, duration = load_audio(str(resolved_path))
    result = analyze_audio_with_progress(
        y,
        sr,
        hop_length=hop_length,
        include_frame_details=include_frame_details,
        progress_callback=progress_callback,
    )
    result["summary"]["duration"] = round(float(duration), 3)
    result["source"] = {
        "filename": Path(path).name,
        "path": str(path),
    }
    return result


def _collect_progress(
    progress_messages: list[str],
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    progress_messages.append(message)
    if progress_callback is not None:
        progress_callback(message)
