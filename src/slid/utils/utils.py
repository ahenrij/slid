"""Utilities."""

import csv
import os
from typing import Dict, List, MutableMapping, Optional


def to_csv(data: List[Dict], output_file: str, mode: Optional[str] = None):
    """Save list of dictionnaries to csv file.
    No effect if data is an empty array.
    """
    if len(data) == 0:
        return
    columns = data[0].keys()
    if mode is None:
        mode = "a" if os.path.isfile(output_file) else "w"

    with open(output_file, mode, newline="") as f:
        dict_writer = csv.DictWriter(f, columns)
        if mode == "w":
            dict_writer.writeheader()
        dict_writer.writerows(data)


def flatten(dictionary, parent_key="", separator="_"):
    """Flatten a nested dictionary."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
