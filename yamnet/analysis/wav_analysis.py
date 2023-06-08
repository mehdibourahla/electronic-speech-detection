import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import json


def process_audio_wrapper(audio_dir, output_path):
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))

    stats = {"failed": 0, "mono_converted": 0, "sample_rates": defaultdict(int)}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_audio, str(file), stats) for file in data}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

    # Convert defaultdict to dict before writing to JSON
    stats["sample_rates"] = dict(stats["sample_rates"])

    with open(output_path, "w") as json_file:
        json.dump(stats, json_file)


def process_audio(path, stats):
    try:
        wav_data, sr = sf.read(path, dtype=np.int16)
        stats["sample_rates"][sr] += 1
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        stats["failed"] += 1
        return None

    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype("float32")

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
        stats["mono_converted"] += 1


def initialize_args(parser):
    # Input paths
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Path to the directory containing audio files",
    )

    parser.add_argument(
        "--output_path", required=True, help="Path to the output the analysis results"
    )


def main(args):
    audio_dir = args.audio_dir
    output_path = args.output_path

    process_audio_wrapper(audio_dir, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
