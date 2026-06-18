from collections import defaultdict
from pathlib import Path
import argparse

from fingerprint import load_audio, fingerprint, save_database, SR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("song_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fingerprints.pkl.gz"))
    args = parser.parse_args()

    paths = sorted(args.song_directory.glob("*.mp3"))
    database = defaultdict(list)
    names = []
    for number, path in enumerate(paths, 1):
        y = load_audio(path)
        duration = len(y) / SR
        max_peaks = min(6500, max(1800, int(duration * 20)))
        _, peaks, hashes = fingerprint(y, max_peaks=max_peaks)
        for key, anchor_time in hashes:
            database[key].append((path.stem, anchor_time))
        names.append(path.stem)
        print(
            f"[{number:02d}/{len(paths)}] {path.stem}: "
            f"{duration:.1f}s, {len(peaks)} peaks, {len(hashes)} hashes"
        )

    save_database(database, names, args.output)
    print(f"Saved {len(database):,} unique hashes to {args.output}")


if __name__ == "__main__":
    main()
