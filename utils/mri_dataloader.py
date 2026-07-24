"""
MRI_Dataloader indexes all samples on disk and hands out DataSample /
Patient / Lesion_Trajectory objects.

DataSample is imported at module level -- that's safe, DataSample never
imports MRI_Dataloader back. Patient and Lesion_Trajectory are imported
lazily inside the methods that use them (iterate_patients,
find_lesion_trajectory, iterate_trajectories) because those two DO import
MRI_Dataloader-adjacent things back (Patient takes a dataloader; the old
Lesion_Trajectory used to build one). Same pattern as lesion_trajectory.py.
"""

import glob
import os

import numpy as np
import plotly.express as px

from .paths import DatasetPaths, DEFAULT_DATA_PATH
from .data_sample import DataSample


class MRI_Dataloader:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH, fast_load: bool = False):
        self.paths = DatasetPaths(data_path)
        self.data_path = data_path
        self.data_prediction_path = self.paths.prediction_root

        if not fast_load:
            globs = glob.glob(self.paths.all_post_scans_glob(), recursive=True)
            self.pre_post_samples = list(set(sorted(globs)))
            self.patient_ids = sorted(list(set([path.split(os.path.sep)[-3] for path in self.pre_post_samples])))

            globs = glob.glob(self.data_prediction_path + "/**/lesion_trajectories_*")
            self.lesion_trajectory_paths = list(set(sorted(globs)))

        self.cache_lesion_trajectories = None

    def __iter__(self):
        for i in range(self.__len__()):
            yield DataSample(self.pre_post_samples[i])

    def __len__(self):
        return len(self.pre_post_samples)

    def find_by_patient_id(self, patient_id):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{patient_id}{os.path.sep}" in sample]
        return [DataSample(sample) for sample in sorted(filtered_samples)]

    def find_by_date(self, date):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{date}{os.path.sep}" in sample]
        return [DataSample(sample) for sample in sorted(filtered_samples)]

    def find_by_patient_id_and_date(self, patient_id, date):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{patient_id}{os.path.sep}" in sample and f"{os.path.sep}{date}{os.path.sep}" in sample]
        return DataSample(filtered_samples[0])

    def find_lesion_trajectory(self, patient_id, label_id):
        from .lesion_trajectory import Lesion_Trajectory  # lazy: see module docstring
        trajectory_path = self.paths.lesion_trajectory_npz_path(patient_id, label_id)
        if os.path.exists(trajectory_path):
            return Lesion_Trajectory(load_from_trajectory_path=trajectory_path)
        else:
            print(f"Lesion trajectory file not found at {trajectory_path}. Cannot load trajectory.")
            return None

    def iterate_patients(self):
        from .patient import Patient  # lazy: see module docstring
        for patient_id in self.patient_ids:
            yield Patient(patient_id, dataloader=self)

    def iterate_trajectories(self):
        from .lesion_trajectory import Lesion_Trajectory  # lazy: see module docstring
        for trajectory_path in self.lesion_trajectory_paths:
            yield Lesion_Trajectory(load_from_trajectory_path=trajectory_path)

    def cache_lesion_trajectories_from_n_scans(self, n_scans: int = 8, only_growing: bool = False):
        trajectories = []
        for trj in self.iterate_trajectories():
            sizes = len(trj.dates)
            if only_growing:
                start_idx, end_idx, _ = trj.extract_growth_phase(trj.sizes)
                sizes = end_idx - start_idx
                trj.allowed_dates = trj.dates[start_idx:end_idx + 1]

            if sizes < n_scans:
                continue

            trajectories.append(trj)

        print(f"Found {len(trajectories)} trajectories with at least {n_scans} scans.")
        self.cache_lesion_trajectories = trajectories

    def plot_lesion_trajectories_line_plot(self, trajectories=None, x_ticks_real_time=False):
        if trajectories is None and self.cache_lesion_trajectories is not None:
            trajectories = self.cache_lesion_trajectories
        elif trajectories is None and self.cache_lesion_trajectories is None:
            print("No trajectories provided and no cached trajectories found. Please provide trajectories or cache them first with self.cache_lesion_trajectories_from_n_scans().")
            return

        fig = px.line()

        for i, trj in enumerate(trajectories):
            from datetime import datetime
            if x_ticks_real_time:
                dates = [datetime.strptime(d, "%Y-%m-%d") for d in trj.dates]
                first_date = dates[0]
                days_since_first = [(d - first_date).days for d in dates]
            else:
                days_since_first = list(range(len(trj.dates)))

            fig.add_scatter(
                x=days_since_first,
                y=np.exp(trj.sizes) ** (1 / 3),
                mode='lines+markers',
                name=f'Patient: {trj.patient_id}, Lesion: {trj.label_id}',
                line=dict(color=px.colors.qualitative.Dark24[i % 24], width=1)
            )

        fig.update_layout(
            title="Lesion Size Over Time",
            xaxis_title="Time",
            yaxis_title="Size",
            yaxis_type="log",
            template="plotly_white",
            height=800
        )

        fig.show()

    def plot_lesion_trajectories_heatmap(self, trajectories=None, skip_top=0, plot_change=False, normalize_rows=False):
        """
        Plot lesion trajectories as a heatmap.

        Args:
            trajectories: List of trajectories to plot. Uses cached trajectories if None.
            skip_top: Number of top trajectories to skip (by max size).
            plot_change: If True, plot change rates instead of absolute sizes.
            normalize_rows: If True, normalize each row to [0, 1].
        """
        if trajectories is None and self.cache_lesion_trajectories is not None:
            trajectories = self.cache_lesion_trajectories
        elif trajectories is None:
            print("No trajectories provided and no cached trajectories found. "
                  "Please provide trajectories or cache them first with self.cache_lesion_trajectories_from_n_scans().")
            return

        trajectories = self._filter_top_trajectories(trajectories, skip_top)

        heatmap_data, trj_list = self._prepare_heatmap_data(trajectories, plot_change, normalize_rows)

        sorted_indices = sorted(range(len(trj_list)), key=lambda i: trj_list[i][1])
        heatmap_data = heatmap_data[sorted_indices]
        trj_list = [trj_list[i] for i in sorted_indices]

        labels = [f'Patient: {trj.patient_id}, Lesion: {trj.label_id} (n={length})'
                  for trj, length in trj_list]

        self._create_heatmap(heatmap_data, labels, plot_change)

    def _filter_top_trajectories(self, trajectories, skip_top):
        """Filter out the top N trajectories by maximum size."""
        trj_with_sizes = []
        for trj in trajectories:
            y = trj.sizes[trj.sizes != 0]
            sizes = np.exp(y) ** (1 / 3)
            trj_with_sizes.append((trj, np.max(sizes)))

        trj_with_sizes.sort(key=lambda x: x[1], reverse=True)
        return [trj for trj, _ in trj_with_sizes[skip_top:]]

    def _prepare_heatmap_data(self, trajectories, plot_change, normalize_rows):
        """Prepare data for heatmap visualization."""
        max_len = max([len(trj.sizes[trj.sizes != 0]) for trj in trajectories])
        if plot_change:
            max_len -= 1

        heatmap_data = []
        trj_list = []

        for trj in trajectories:
            y = trj.sizes[trj.sizes != 0]
            sizes = np.exp(y) ** (1 / 3)

            if plot_change:
                data = np.diff(sizes)
            else:
                data = sizes

            padded = np.full(max_len, np.nan)
            padded[:len(data)] = data

            if normalize_rows:
                padded = self._normalize_row(padded)

            heatmap_data.append(padded)
            trj_list.append((trj, len(sizes)))

        return np.array(heatmap_data), trj_list

    def _normalize_row(self, row):
        """Normalize a row to [0, 1]."""
        valid_mask = ~np.isnan(row)
        if not np.any(valid_mask):
            return row

        min_val = np.nanmin(row)
        max_val = np.nanmax(row)

        if max_val > min_val:
            row[valid_mask] = (row[valid_mask] - min_val) / (max_val - min_val)
        else:
            row[valid_mask] = 0.5

        return row

    def _create_heatmap(self, heatmap_data, labels, plot_change):
        """Create and display the heatmap."""
        max_len = heatmap_data.shape[1]

        if plot_change:
            color_scale = "RdBu_r"
            color_label = "Change Rate"
            max_abs = np.nanmax(np.abs(heatmap_data))
            zmin, zmax = -max_abs, max_abs
        else:
            color_scale = "Viridis"
            color_label = "Size"
            zmin, zmax = None, None

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Time", y="Trajectory", color=color_label),
            x=list(range(max_len)),
            y=labels,
            color_continuous_scale=color_scale,
            title=f"Lesion {'Change Rates' if plot_change else 'Size'} Over Time (Heatmap - sorted by length)",
            zmin=zmin,
            zmax=zmax
        )

        fig.update_layout(height=max(600, len(labels) * 15), width=900)
        fig.show()