from pathlib import Path
import io

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fingerprint import HOP, N_FFT, SR, identify, load_audio, load_database


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "fingerprints.pkl.gz"

st.set_page_config(
    page_title="Sonic Signatures",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f4ef; color: #17201f; }
    [data-testid="stSidebar"] { background: #17201f; }
    [data-testid="stSidebar"] * { color: #f6f4ef; }
    .hero {
        border: 1px solid #17201f; padding: 1.5rem 1.7rem;
        background: #fffdf8; box-shadow: 7px 7px 0 #d7e4de;
        margin-bottom: 1.2rem;
    }
    .eyebrow { font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }
    .result {
        border-left: 7px solid #d96c3f; background: #fffdf8;
        padding: 1rem 1.3rem; margin: 1rem 0;
    }
    .metric-card {
        background:#fffdf8; border:1px solid #a7aaa4; padding:.75rem 1rem;
        min-height:92px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 0; border: 1px solid #17201f; background: #d7e4de;
        color: #17201f; font-weight: 700;
    }
    div[data-testid="stFileUploader"] { background:#fffdf8; padding:.5rem; border:1px dashed #65736e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the 50-song fingerprint library...")
def database():
    return load_database(DATABASE_PATH)


def spectrogram_figure(result):
    times = librosa.frames_to_time(
        np.arange(result.spectrogram_db.shape[1]), sr=SR, hop_length=HOP
    )
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    mesh = ax.pcolormesh(
        times, frequencies, result.spectrogram_db,
        shading="auto", cmap="magma", vmin=-70, vmax=0
    )
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 5000))
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    return fig


def constellation_figure(result):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if result.peaks:
        points = np.asarray(result.peaks)
        times = librosa.frames_to_time(points[:, 0], sr=SR, hop_length=HOP)
        frequencies = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[points[:, 1]]
        ax.scatter(times, frequencies, s=14, color="#1f6f78", alpha=0.85)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 5000))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def offset_figure(result):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if result.offset_histogram:
        offsets = np.array(list(result.offset_histogram)) * HOP / SR
        counts = np.array(list(result.offset_histogram.values()))
        order = np.argsort(offsets)
        ax.bar(offsets[order], counts[order], width=HOP / SR * 0.9, color="#d96c3f")
    ax.axvline(result.offset_frames * HOP / SR, color="#17201f", linestyle="--", linewidth=1)
    ax.set(xlabel="Database time - query time (s)", ylabel="Matching hash votes")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def analyse(upload):
    suffix = Path(upload.name).suffix.lower() or ".mp3"
    y = load_audio(upload, suffix=suffix)
    return identify(y, database())


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">EE200 · Signals, Systems and Networks</div>
      <h1 style="margin:.25rem 0 .35rem 0">Sonic Signatures</h1>
      <div>A compact audio-fingerprinting lab that identifies a song from a short clip.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose a workflow", ["Single clip", "Batch clips"], label_visibility="collapsed")
    st.divider()
    payload = database()
    st.metric("Indexed songs", len(payload["songs"]))
    st.caption("Predictions use each song's original filename without the extension.")

if mode == "Single clip":
    st.subheader("Identify one clip")
    upload = st.file_uploader(
        "Upload a query clip",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        accept_multiple_files=False,
    )
    if upload:
        st.audio(upload)
        if st.button("Analyse clip", type="primary", use_container_width=True):
            with st.spinner("Extracting peaks and aligning fingerprints..."):
                result = analyse(upload)
            st.session_state["single_result"] = result
            st.session_state["single_name"] = upload.name

    result = st.session_state.get("single_result")
    if result:
        st.markdown(
            f"""
            <div class="result">
              <div class="eyebrow">Predicted song</div>
              <h2 style="margin:.25rem 0">{result.prediction}</h2>
              <div>Strongest aligned offset: <b>{result.score}</b> votes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        columns = st.columns(4)
        values = [
            ("Clip duration", f"{result.duration:.1f} s"),
            ("Constellation peaks", f"{len(result.peaks):,}"),
            ("Paired hashes", f"{result.hashes_count:,}"),
            ("Winning margin", f"{result.score - result.second_score}"),
        ]
        for column, (label, value) in zip(columns, values):
            column.markdown(
                f'<div class="metric-card"><small>{label}</small><h3>{value}</h3></div>',
                unsafe_allow_html=True,
            )

        st.subheader("1. Spectrogram")
        st.pyplot(spectrogram_figure(result), use_container_width=True)
        st.subheader("2. Constellation map")
        st.pyplot(constellation_figure(result), use_container_width=True)
        st.subheader("3. Offset histogram")
        st.pyplot(offset_figure(result), use_container_width=True)
        with st.expander("Candidate scores"):
            st.dataframe(
                pd.DataFrame(result.ranked, columns=["song", "aligned_offset_votes"]),
                hide_index=True,
                use_container_width=True,
            )

else:
    st.subheader("Identify multiple clips")
    st.write(
        "Upload several query clips. The downloaded file contains exactly two columns: "
        "`filename,prediction`."
    )
    uploads = st.file_uploader(
        "Upload query clips",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Run batch", type="primary", use_container_width=True):
        rows = []
        progress = st.progress(0, text="Starting batch...")
        for index, upload in enumerate(uploads, 1):
            progress.progress(
                index / len(uploads), text=f"Identifying {upload.name}..."
            )
            result = analyse(upload)
            rows.append(
                {
                    "filename": upload.name,
                    "prediction": (
                        result.prediction
                        if result.prediction == "No confident match"
                        else Path(result.prediction).stem
                    ),
                }
            )
        progress.empty()
        st.session_state["batch_results"] = pd.DataFrame(
            rows, columns=["filename", "prediction"]
        )

    batch = st.session_state.get("batch_results")
    if batch is not None:
        st.success(f"Finished {len(batch)} clip(s).")
        st.dataframe(batch, hide_index=True, use_container_width=True)
        csv_bytes = batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )
