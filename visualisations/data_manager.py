"""File-system I/O, caching, and frame iteration."""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import fsspec
import networkx as nx
import pandas as pd


class DataManager:
    """Encapsulates file-system operations and frame iteration."""

    def __init__(self, root: Path, seed: int = 42):
        self.root = Path(root)
        self._frame_cache: dict = {}
        self._csv_cache: dict = {}
        random.seed(seed)

    # ── Directory helpers ────────────────────────────────────────────────────

    def list_dir(self, path: Path) -> List[str]:
        """Sorted list of non-hidden entries inside *path*."""
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if not p.name.startswith("."))

    def list_leaf_dirs(self, folder: Path) -> List[str]:
        """Return relative paths of all leaf directories under *folder*."""
        folder = Path(folder)
        return [
            str(p.relative_to(folder))
            for p in folder.rglob("*")
            if p.is_dir() and not any(c.is_dir() for c in p.iterdir())
        ]

    # ── Progress widget helpers ──────────────────────────────────────────────

    @staticmethod
    def _set_progress(widget_source, visible: bool, active: bool = False, status: str = ""):
        """Set progress indicators if available.
        
        Args:
            widget_source: Object that may have .w.progress and .w.status attributes
                          (widget manager is accessed via .w for plot instances)
        """
        if widget_source is None:
            return
        
        # Handle both direct widget_manager and plot instances (which have .w attribute)
        widgets = getattr(widget_source, 'w', widget_source)
        if not hasattr(widgets, 'progress'):
            return
            
        widgets.progress.visible = visible
        widgets.progress.active = active
        widgets.status.visible = visible
        if status:
            widgets.status.object = status

    # ── Reservoir sampling ───────────────────────────────────────────────────

    @staticmethod
    def _reservoir_sample(
        fh,
        max_frames: int,
        sample_range: Tuple[int, int],
        widget_mgr=None,
        data_type: str = "lines",
    ) -> list:
        sample: list = []
        start, end = sample_range

        for i, line in enumerate(fh, start=1):
            if i < start:
                continue
            if i > end:
                break
            if len(sample) < max_frames:
                sample.append(line)
            else:
                j = random.randint(1, i)
                if j <= max_frames:
                    sample[j - 1] = line
            if widget_mgr:
                # Handle both direct widget_manager and plot instances (which have .w attribute)
                widgets = getattr(widget_mgr, 'w', widget_mgr)
                if hasattr(widgets, 'status'):
                    widgets.status.object = f"**{i:,} {data_type} processed...**"

        return sample

    # ── Frame loading ────────────────────────────────────────────────────────

    def load_frames(
        self,
        file_path: Path,
        widget_source=None,
    ) -> Iterable[nx.Graph]:
        """Yield NetworkX graphs from a zstd-compressed JSON-lines file.
        
        Args:
            file_path: Path to the graph file
            widget_source: Object with max_frames_input, range_start, range_end attributes
                          (can be a plot instance or widget manager)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return

        max_frames = max(1, widget_source.max_frames_input.value) if widget_source else 10_000
        if widget_source:
            start = max(0, widget_source.range_start.value or 0)
            end = max(start + 1, widget_source.range_end.value or float("inf"))
            sample_range = (start, end)
        else:
            sample_range = (0, float("inf"))
        
        cache_key = (file_path, max_frames, sample_range)
        if cache_key in self._frame_cache:
            yield from self._frame_cache[cache_key]
            return

        self._set_progress(widget_source, visible=True, active=True, status="**Loading JSON frames...**")
        try:
            with fsspec.open(file_path, "rt", compression="zstd") as f:
                sample = self._reservoir_sample(f, max_frames, sample_range, widget_source, data_type="JSON frames")
            frames = [
                nx.readwrite.json_graph.node_link_graph(json.loads(line), edges="links")
                for line in sample
            ]
            self._frame_cache[cache_key] = frames
            yield from frames
        finally:
            self._set_progress(widget_source, visible=False)

    def get_frame(self, file_path: Path, idx: int, widget_source=None) -> Optional[nx.Graph]:
        """Return the *idx*-th frame from *file_path*."""
        for i, frame in enumerate(self.load_frames(file_path, widget_source)):
            if i == idx:
                return frame
        return None

    def count_frames(self, file_path: Path, widget_source=None) -> int:
        return sum(1 for _ in self.load_frames(file_path, widget_source))

    # ── CSV loading ──────────────────────────────────────────────────────────

    def read_csv(self, path: Path, widget_source=None) -> pd.DataFrame:
        """Read a zstd-compressed CSV with optional progress updates.
        
        Args:
            path: Path to the CSV file
            widget_source: Object with max_frames_input, range_start, range_end attributes
                          (can be a plot instance or widget manager)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        max_frames = max(1, widget_source.max_frames_input.value) if widget_source else 10_000
        if widget_source:
            start = max(0, widget_source.range_start.value or 0)
            end = max(start + 1, widget_source.range_end.value or float("inf"))
            sample_range = (start, end)
        else:
            sample_range = (0, float("inf"))
        
        cache_key = (path, max_frames, sample_range)

        if cache_key in self._csv_cache:
            header, sample = self._csv_cache[cache_key]
        else:
            self._set_progress(widget_source, visible=True, active=True, status="**Loading CSV data...**")
            try:
                # Optimization: if max_frames can fit all rows in range, load directly with pandas
                if end != float("inf") and max_frames >= (end - start):
                    # Read the range directly without reservoir sampling
                    df = pd.read_csv(
                        path,
                        skiprows=start,
                        nrows=end - start,
                        compression="zstd"
                    )
                    self._set_progress(widget_source, visible=False)
                    return df.sort_values(by="timestamp")
                
                # Otherwise use reservoir sampling for large files
                with fsspec.open(path, "rt", compression="zstd") as f:
                    header = next(f)
                    sample = self._reservoir_sample(f, max_frames, sample_range, widget_source, data_type="CSV rows")
                self._csv_cache[cache_key] = (header, sample)
            finally:
                self._set_progress(widget_source, visible=False)

        df = pd.read_csv(io.StringIO(header + "".join(sample)))
        return df.sort_values(by="timestamp")
