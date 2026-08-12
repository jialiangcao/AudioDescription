"""Timestamped frame lookup over a Timeline.

Symphony's tools located frames by directory listing + a hardcoded 2fps
index-to-seconds conversion. Here every sampled frame already carries an
authoritative absolute timestamp (``Frame.time``, set by segmentation), so the
tools work against this index instead of doing fps arithmetic.
"""

import bisect
import logging
from dataclasses import dataclass, field

import numpy as np

from timeline import Timeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameIndex:
    # (timestamp_sec, blob_key) sorted by timestamp.
    entries: list[tuple[float, str]]
    duration_sec: float
    _time_by_key: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_time_by_key", {key: ts for ts, key in self.entries})

    @classmethod
    def from_timeline(cls, timeline: Timeline) -> "FrameIndex":
        entries = sorted(
            (frame.time, frame.key) for seg in timeline.segments for frame in seg.frames
        )
        logger.debug(
            "FrameIndex: %d frame(s) over %.1fs", len(entries), timeline.duration_sec
        )
        return cls(entries=entries, duration_sec=timeline.duration_sec)

    def keys(self) -> list[str]:
        return [key for _, key in self.entries]

    def timestamp_of(self, key: str) -> float:
        return self._time_by_key[key]

    def in_range(self, start_sec: float, end_sec: float) -> list[tuple[float, str]]:
        """Frames with start_sec <= timestamp <= end_sec, in temporal order."""
        timestamps = [ts for ts, _ in self.entries]
        lo = bisect.bisect_left(timestamps, start_sec)
        hi = bisect.bisect_right(timestamps, end_sec)
        return self.entries[lo:hi]

    def windows(self, window_sec: float) -> list[tuple[float, float, list[str]]]:
        """Bucket every frame into a fixed [k*w, (k+1)*w) grid over the video.

        Returns (window_start, window_end, frame_keys) per non-empty window,
        in temporal order — the replacement for Symphony's group_frames.
        """
        buckets: dict[int, list[str]] = {}
        for ts, key in self.entries:
            buckets.setdefault(int(ts // window_sec), []).append(key)
        return [
            (k * window_sec, (k + 1) * window_sec, buckets[k]) for k in sorted(buckets)
        ]

    def uniform_sample(self, start_sec: float, end_sec: float, count: int) -> list[str]:
        """Up to ``count`` frames spread uniformly across [start_sec, end_sec).

        Sample instants come from linspace(endpoint=False), mirroring
        Symphony's uniform selection; each instant maps to the nearest frame
        within the range, then duplicates collapse (temporal order preserved).
        """
        if count <= 0:
            return []
        candidates = self.in_range(start_sec, end_sec)
        if not candidates:
            return []
        timestamps = [ts for ts, _ in candidates]
        picked: list[str] = []
        for instant in np.linspace(start_sec, end_sec, num=count, endpoint=False):
            pos = bisect.bisect_left(timestamps, float(instant))
            if (
                pos == len(timestamps)
                or pos > 0
                and abs(timestamps[pos - 1] - instant) <= abs(timestamps[pos] - instant)
            ):
                pos -= 1
            key = candidates[pos][1]
            if key not in picked:
                picked.append(key)
        return picked
