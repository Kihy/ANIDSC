# Intrusion Detection Dashboard

Interactive Panel dashboard for exploring multilayer network intrusion-detection data.

## Project layout

```
dashboard/
├── main.py              # Entry point — run this
├── config.py            # Constants (ROOT path, MAC → device map, layer colours)
├── data_manager.py      # File I/O, reservoir sampling, frame/CSV caching
├── widget_manager.py    # Shared Panel widgets + path breadcrumb navigator
├── plot_manager.py      # Wires the plot registry into sidebar & container
└── plots/
    ├── __init__.py      # ← Register new plots here (1 line)
    ├── base.py          # BasePlot ABC every plot must subclass
    ├── node_plot.py     # Node anomaly-score time series
    ├── network_plot.py  # Interactive multilayer graph snapshot
    ├── scores_plot.py   # Score distribution (median / IQR / range)
    └── summary_plot.py  # Cross-pipeline accuracy / F1 / mAP bar charts
```

## Running

```bash
panel serve main.py --address 0.0.0.0 --port 5007 --dev

# or
python main.py                      # single-process
```

## Adding a new plot (2 steps)

1. **Create** `plots/my_plot.py`:

```python
from pathlib import Path
import panel as pn
from plots.base import BasePlot

class MyPlot(BasePlot):
    label = "My Plot"                        # appears on the mode dropdown
    description = "What this plot shows"
    file_type = "csv"                        # "graph" for JSON or "csv" for CSV files

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        # Return whatever Panel widgets this mode needs.
        # self.w exposes all shared widgets (max_frames_input, etc.)
        return pn.WidgetBox("Parameters", self.w.max_frames_input)

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        # Build and return a Panel pane.
        # self.data_mgr gives access to load_frames (graph) / read_csv (csv).
        if self.file_type == "csv":
            df = self.data_mgr.read_csv(selected_path, self.w)
        else:  # graph
            frames = list(self.data_mgr.load_frames(selected_path, self.w))
        ...
        return pn.pane.Plotly(fig)
```

2. **Done!** The plot is automatically discovered and appears in the dropdown.

The dashboard automatically:
- Discovers any class ending with "Plot" in the plots/ directory
- Loads the correct file type (graph JSON or CSV) based on `file_type`
- Populates appropriate widgets when files are selected

## Shared widgets available via `self.w`

| Widget | Type | Used by |
|---|---|---|
| `max_frames_input` | IntInput | Node, Network, Scores |
| `range_start` / `range_end` | IntInput | Node, Network, Scores |
| `node_input` | Select | Node |
| `graph_slider` | EditableIntSlider | Network |
| `metric_input` | Select | Summary |
| `pipeline_regex` | TextInput | Summary |
| `plot_height` | IntInput | Summary |
| `progress` / `status` | Progress/Markdown | all (auto-managed by DataManager) |
