from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import gzip
import pickle
import tempfile

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


SR = 11025
N_FFT = 2048
HOP = 512
PEAK_NEIGHBORHOOD = (15, 7)
PEAK_THRESHOLD_DB = -45.0
PEAK_PERCENTILE = 91
QUERY_MAX_PEAKS = 600
FANOUT = 8
MIN_DT = 2
MAX_DT = 65


@dataclass
class MatchResult:
    prediction: str
    score: int
    second_score: int
    offset_frames: int
    ranked: list[tuple[str, int]]
    offset_histogram: Counter
    spectrogram_db: np.ndarray
    peaks: list[tuple[int, int]]
    hashes_count: int
    duration: float


def load_audio(source, suffix: str = ".mp3") -> np.ndarray:
    if isinstance(source, (str, Path)):
        y, _ = librosa.load(str(source), sr=SR, mono=True)
    else:
        data = source.getvalue() if hasattr(source, "getvalue") else source.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        finally:
            Path(path).unlink(missing_ok=True)
    if y.size and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)


def spectrogram_db(y: np.ndarray) -> np.ndarray:
    magnitude = np.abs(
        librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann")
    )
    return librosa.amplitude_to_db(magnitude + 1e-10, ref=np.max)


def find_peaks(s_db: np.ndarray, max_peaks: int = QUERY_MAX_PEAKS):
    local_max = s_db == maximum_filter(
        s_db, size=PEAK_NEIGHBORHOOD, mode="nearest"
    )
    threshold = max(PEAK_THRESHOLD_DB, float(np.percentile(s_db, PEAK_PERCENTILE)))
    coordinates = np.argwhere(local_max & (s_db >= threshold))
    if len(coordinates) > max_peaks:
        strengths = s_db[coordinates[:, 0], coordinates[:, 1]]
        coordinates = coordinates[np.argsort(strengths)[-max_peaks:]]
    peaks = [(int(time), int(freq)) for freq, time in coordinates]
    return sorted(peaks)


def make_hashes(peaks, fanout: int = FANOUT):
    hashes = []
    for index, (time_1, freq_1) in enumerate(peaks):
        partners = 0
        for time_2, freq_2 in peaks[index + 1 :]:
            delta_time = time_2 - time_1
            if delta_time < MIN_DT:
                continue
            if delta_time > MAX_DT:
                break
            key = (freq_1 // 2, freq_2 // 2, delta_time)
            hashes.append((key, time_1))
            partners += 1
            if partners >= fanout:
                break
    return hashes


def fingerprint(y: np.ndarray, max_peaks: int = QUERY_MAX_PEAKS):
    s_db = spectrogram_db(y)
    peaks = find_peaks(s_db, max_peaks=max_peaks)
    hashes = make_hashes(peaks)
    return s_db, peaks, hashes


def save_database(database, song_names, output_path):
    payload = {
        "version": 1,
        "parameters": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop": HOP,
            "neighborhood": PEAK_NEIGHBORHOOD,
            "threshold_db": PEAK_THRESHOLD_DB,
            "percentile": PEAK_PERCENTILE,
            "fanout": FANOUT,
            "min_dt": MIN_DT,
            "max_dt": MAX_DT,
        },
        "songs": sorted(song_names),
        "database": dict(database),
    }
    with gzip.open(output_path, "wb", compresslevel=6) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_database(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def identify(y: np.ndarray, payload) -> MatchResult:
    s_db, peaks, hashes = fingerprint(y)
    votes = defaultdict(Counter)
    for key, query_time in hashes:
        for song, database_time in payload["database"].get(key, ()):
            votes[song][database_time - query_time] += 1

    ranked = sorted(
        (
            (song, max(histogram.values()) if histogram else 0)
            for song, histogram in votes.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if ranked:
        prediction, score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        best_offset, _ = votes[prediction].most_common(1)[0]
        histogram = votes[prediction]
    else:
        prediction, score, second_score, best_offset = "No confident match", 0, 0, 0
        histogram = Counter()

    return MatchResult(
        prediction=prediction,
        score=score,
        second_score=second_score,
        offset_frames=best_offset,
        ranked=ranked[:8],
        offset_histogram=histogram,
        spectrogram_db=s_db,
        peaks=peaks,
        hashes_count=len(hashes),
        duration=len(y) / SR,
    )
