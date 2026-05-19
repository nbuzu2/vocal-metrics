import numpy as np
from scipy.signal import find_peaks, welch


def analyze_vibrato(df, sr, hop_length=512):
    sustained = df[df["rms"] > df["rms"].quantile(0.7)]["frequency_smoothed"].dropna()

    if len(sustained) < 50:
        return {"vibrato_rate_hz": 0, "vibrato_extent_hz": 0}

    diff = np.diff(sustained.values)
    peaks = len(find_peaks(diff)[0]) + len(find_peaks(-diff)[0])
    duration = len(sustained) * hop_length / sr

    return {
        "vibrato_rate_hz": peaks / (2 * duration),
        "vibrato_extent_hz": float(np.std(sustained)),
    }


def analyze_pitch_modulation(df, sr, hop_length=512):
    """
    Descompone la trayectoria de pitch en tres bandas de modulación usando análisis espectral
    (densidad espectral de potencia via método de Welch).

    La idea central es que cualquier oscilación del pitch tiene una frecuencia característica:
    qué tan rápido sube y baja el pitch por segundo. Según esa velocidad, la oscilación
    tiene un nombre musical y un significado diferente:

    WOBBLE (0.5 – 3.5 Hz)
        Oscilación lenta e involuntaria. El cantante no puede sostener la nota con firmeza
        y el pitch "balancea" de forma irregular. Es lo que se llama "voz temblorosa" o
        "pitch inestable". No es deseable. Común en cantantes sin entrenamiento o con
        tensión excesiva.

    VIBRATO (3.5 – 8.0 Hz)
        Oscilación rápida y controlada, producida voluntariamente como técnica expresiva.
        El rango ideal en canto clásico y lírico es 5–7 Hz. Le da "vida" a la nota sostenida.
        Es deseable y esperado en estilos formales. En pop/contemporáneo se usa con menos
        frecuencia o de forma más sutil.

    FLUTTER (8.0 – 15.0 Hz)
        Oscilación muy rápida, más allá del vibrato natural. Generalmente involuntaria.
        Puede deberse a tensión laríngea, fatiga vocal o una técnica de "trill" muy cerrada.
        No es deseable en contextos de canto sostenido.

    Método:
        1. Se toma la trayectoria de frecuencia fundamental (frequency_smoothed) de los
           frames con voz.
        2. Se elimina la tendencia melódica lenta (~0.5s de ventana) mediante convolución,
           dejando solo las modulaciones de corto plazo.
        3. Se aplica el método de Welch para estimar la densidad espectral de potencia (PSD)
           de esas modulaciones.
        4. Se calcula qué proporción de la energía total cae en cada banda.

    wobble_score (0–100):
        Mide qué tan libre de wobble está el cantante. 100 = sin wobble, 0 = todo wobble.
        Se penaliza directamente con la proporción de energía en la banda de wobble.
    """
    voiced = df[df["is_voice"]]["frequency_smoothed"].dropna()
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]

    frame_rate = sr / hop_length
    if len(voiced) < int(frame_rate * 1.5):
        return {
            "wobble_score": 100.0,
            "wobble_energy_ratio": 0.0,
            "vibrato_energy_ratio": 0.0,
            "flutter_energy_ratio": 0.0,
        }

    pitch = voiced.values.astype(float)

    # Remove slow melodic trend (~0.5s moving average) to isolate modulations
    trend_window = max(3, int(frame_rate * 0.5))
    trend = np.convolve(pitch, np.ones(trend_window) / trend_window, mode="same")
    modulation = pitch - trend

    nperseg = min(len(modulation), max(int(frame_rate * 2), 64))
    freqs, psd = welch(modulation, fs=frame_rate, nperseg=nperseg)

    base_mask = freqs >= 0.5
    total = psd[base_mask].sum()
    if total == 0:
        return {
            "wobble_score": 100.0,
            "wobble_energy_ratio": 0.0,
            "vibrato_energy_ratio": 0.0,
            "flutter_energy_ratio": 0.0,
        }

    wobble_ratio = float(psd[(freqs >= 0.5) & (freqs < 3.5)].sum() / total)
    vibrato_ratio = float(psd[(freqs >= 3.5) & (freqs < 8.0)].sum() / total)
    flutter_ratio = float(psd[(freqs >= 8.0) & (freqs < 15.0)].sum() / total)

    # Regularity: how concentrated is the vibrato energy around a single frequency.
    # A pure, controlled vibrato creates a sharp peak; wobble/irregular oscillation
    # spreads energy across the band. Measured as the share of vibrato-band energy
    # within ±0.5 Hz of the dominant frequency.
    vib_mask = (freqs >= 3.5) & (freqs < 8.0)
    if vib_mask.any() and psd[vib_mask].sum() > 0:
        vib_freqs = freqs[vib_mask]
        peak_freq = float(vib_freqs[np.argmax(psd[vib_mask])])
        near_peak = (freqs >= peak_freq - 0.5) & (freqs <= peak_freq + 0.5) & vib_mask
        vibrato_regularity = round(float(psd[near_peak].sum() / psd[vib_mask].sum() * 100), 1)
        dominant_vibrato_hz = round(peak_freq, 2)
    else:
        vibrato_regularity = 0.0
        dominant_vibrato_hz = 0.0

    # 0% wobble energy → 100, 100% wobble energy → 0
    wobble_score = round(max(0.0, 100.0 - wobble_ratio * 100.0), 1)

    return {
        "wobble_score": wobble_score,
        "wobble_energy_ratio": round(wobble_ratio, 3),
        "vibrato_energy_ratio": round(vibrato_ratio, 3),
        "flutter_energy_ratio": round(flutter_ratio, 3),
        "vibrato_regularity": vibrato_regularity,
        "dominant_vibrato_hz": dominant_vibrato_hz,
    }

