#!/usr/bin/env python
"""
A util script for dumping values logged in a TensorFlow events file.

Usage:
    ./dump_tfevents.py <path/to/tfevents/file>
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from tensorflow.core.util import event_pb2
from tensorflow.data import TFRecordDataset
# from tensorflow.python.summary.summary_iterator import summary_iterator


def main():
    path = Path(sys.argv[1])
    if path.is_dir():
        for file in path.iterdir():
            if "tfevents" in file.name:
                path = file
                break
    assert path.is_file(), f"tfevents file not found: {path}"

    # for event in summary_iterator(str(path)):
    for serialized_event in TFRecordDataset(str(path)):
        event = event_pb2.Event.FromString(serialized_event.numpy())
        for v in event.summary.value:
            try:
                simple_value = v.simple_value
                step = event.step
            except AttributeError as e:
                print(e)
                continue
            timestamp = datetime.fromtimestamp(event.wall_time, timezone.utc)
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{step}\t{v.tag:>16}: {simple_value:>7.4f} \t ({time_str})")


if __name__ == "__main__":
    main()
