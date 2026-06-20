# Sonic Signatures - EE200 Q3B.

A Streamlit audio-fingerprinting app for identifying short song clips.

## Features

- Single-clip identification
- Spectrogram, constellation map, and offset histogram
- Batch identification
- Exact `results.csv` columns: `filename,prediction`
- Precomputed 50-song fingerprint database

## Run locally.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `fingerprints.pkl.gz` in the same directory as `app.py`.

## Rebuild the fingerprint database

```bash
python build_database.py /path/to/song_library --output fingerprints.pkl.gz
```

Song filenames are preserved as prediction labels without their extensions.
