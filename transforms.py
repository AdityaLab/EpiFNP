"""Transform functions for time series"""
from typing import List, Tuple
import numpy as np


def create_mask(lens, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[:j, i, :] = 1.0
    return ans


def create_mask2(lens, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[j - 1, i, :] = 1.0
    return ans


def seqs_to_arrs(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a list of sequences to a numpy array"""
    lens = [len(seq) for seq in seqs]
    max_len = max(lens)
    ans = np.zeros((len(seqs), max_len), dtype="float32")
    for i, seq in enumerate(seqs):
        ans[i, : lens[i]] = seq
    return ans, create_mask(lens)


def shift_start(
    series: List[np.ndarray], target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Shifts the start of the series to the target"""
    ans = []
    for ts in series:
        # Find the index with value larger than target[0]
        idx = np.argmax(ts > target[0])
        # Shift the series
        ans.append(ts[idx:])
    return seqs_to_arrs(ans)


def scale_to_max(
    series: List[np.ndarray], target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Scales the series to have max value 1"""
    ans = []
    for ts in series:
        ans.append(ts / np.max(ts)) * target.max()
    return seqs_to_arrs(ans)
