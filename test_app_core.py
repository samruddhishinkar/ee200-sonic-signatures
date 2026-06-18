from pathlib import Path
import io

import pandas as pd
import soundfile as sf

from fingerprint import identify, load_audio, load_database, SR


ROOT = Path(__file__).resolve().parents[1]
SONGS = ROOT / "work" / "songs"
DATABASE = load_database(Path(__file__).with_name("fingerprints.pkl.gz"))


def clip_bytes(song, start, duration=8):
    path = SONGS / f"{song}.mp3"
    audio = load_audio(path)
    start_sample = int(start * SR)
    clip = audio[start_sample : start_sample + int(duration * SR)]
    buffer = io.BytesIO()
    sf.write(buffer, clip, SR, format="WAV")
    return buffer.getvalue()


def main():
    cases = [
        ("Yesterday", 35),
        ("Under Pressure", 120),
        ("We Are The Champions", 100),
        ("Bohemian Rhapsody", 180),
        ("A Day In The Life", 250),
    ]
    rows = []
    for song, start in cases:
        audio = load_audio(io.BytesIO(clip_bytes(song, start)), suffix=".wav")
        result = identify(audio, DATABASE)
        rows.append(
            {
                "filename": f"{song}_query",
                "prediction": result.prediction,
                "expected": song,
                "score": result.score,
            }
        )
        assert result.prediction == song, rows[-1]

    csv = pd.DataFrame(rows)[["filename", "prediction"]].to_csv(index=False)
    assert csv.splitlines()[0] == "filename,prediction"
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nCSV preview:\n" + csv)


if __name__ == "__main__":
    main()
