"""
Lesion_Trajectory tracks a single lesion's size across scans over time.

Note the two lazy `from patient import Patient` imports below. This is a
*genuine* mutual dependency (Patient.load_lesion_trajectories() builds
Lesion_Trajectory objects, and Lesion_Trajectory needs Patient.samples) --
not just a shared path string -- so a module-level import would create a
real cycle with patient.py. Importing inside the method body is the
standard, idiomatic way to break that: the import only runs when the
method is actually called, by which point both modules have finished
loading.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from IPython.display import HTML
from scipy.signal import savgol_filter
from datetime import datetime

from .paths import DatasetPaths, DEFAULT_DATA_PATH


class Lesion_Trajectory:
    def __init__(self, patient_id=None, label_id=None, sample_ids: list = None, sizes=None,
                 load_from_trajectory_path=None, data_path: str = DEFAULT_DATA_PATH):
        self.paths = DatasetPaths(data_path)

        if load_from_trajectory_path is not None:
            self.path = load_from_trajectory_path
            self.load_lesion_trajectory()
        else:
            # Previously: MRI_Dataloader(fast_load=True).data_prediction_path
            # -- that instantiated an entire dataloader just to read one
            # string. DatasetPaths gives us the same string with no
            # dependency on MRI_Dataloader at all.
            self.path = self.paths.lesion_trajectory_npz_path(patient_id, label_id)
            self.patient_id = patient_id
            self.label_id = label_id
            self.dates = []

            from patient import Patient  # lazy: genuine mutual dependency, see module docstring
            patient = Patient(patient_id)
            for i, sample in enumerate(patient.samples):
                if i not in sample_ids:
                    continue
                sample.load_mri_segmentation()
                self.dates.append(sample.date)
            self.n_scans = len(sample_ids)
            self.sizes = sizes

    def save_lesion_trajectory(self):
        export = {
            "patient_id": self.patient_id,
            "label_id": self.label_id,
            "dates": self.dates,
            "n_scans": self.n_scans,
            "sizes": self.sizes,
        }

        np.savez(self.path, **export)

    def load_lesion_trajectory(self):
        if os.path.exists(self.path):
            data = np.load(self.path)
            self.patient_id = data["patient_id"].item()
            self.label_id = data["label_id"].item()
            self.dates = data["dates"].tolist()
            self.n_scans = data["n_scans"].item()
            self.sizes = data["sizes"]
        else:
            print(f"Lesion trajectory file not found at {self.path}. Cannot load trajectory.")

    def load_sizes(self):
        sizes = []

        from patient import Patient  # lazy: see module docstring
        patient = Patient(self.patient_id)
        for i, date in enumerate(patient.dates):
            if date not in self.dates:
                continue

            sample = patient.samples[i]
            data, num_features = sample.load_lesion_trajectory_segmentation()
            size = np.sum(data == self.label_id)
            sizes.append(size)
        self.sizes = sizes
        return sizes

    def load_labels_for_inr(self, selected_date=None, absolute_day_number=False, skip_empty=True):
        """
        Load trajectory labels for each date.

        Args:
            selected_date: If provided, only load this specific date
            absolute_day_number: Use absolute days or normalized time
            skip_empty: If True, skip frames with no voxels (helpful for trajectories with gaps)
        """
        dates_allowed = self.allowed_dates if hasattr(self, 'allowed_dates') else self.dates


        dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates_allowed]
        first_date = dates[0]
        total_days = (dates[-1] - first_date).days if not absolute_day_number else 1
        days_since_first = [((d - first_date).days / total_days) for d in dates]
        if not absolute_day_number:
            days_since_first = [d * 2 - 1 for d in days_since_first]

        data = []
        skipped_count = 0

        for i, date in enumerate(dates_allowed):
            if selected_date is not None:
                date = selected_date
                i = dates_allowed.index(selected_date)

            path = os.path.join(*self.path.split(os.path.sep)[:-1] + [date] + ["trajectory.nii.gz"])
            if os.path.exists(path):
                data_segmentation = nib.load(path).get_fdata()
                data_segmentation = np.where(data_segmentation == self.label_id, 1, 0).astype(np.uint8)

                voxel_count = np.count_nonzero(data_segmentation)

                if skip_empty and voxel_count == 0:
                    skipped_count += 1
                    if selected_date is None:
                        continue
                    data.append((data_segmentation, days_since_first[i]))
                    if selected_date is not None:
                        return data[0]
                else:
                    data.append((data_segmentation, days_since_first[i]))
                    if selected_date is not None:
                        return data[0]
            else:
                if selected_date is not None:
                    print(f"Path not found: {path}")
                continue

        if selected_date is None and skipped_count > 0:
            print(f"Note: Skipped {skipped_count} empty frames from trajectory")

        return data

    def plot_trajectory_sizes(self):
        fig, ax = plt.subplots()
        ax.plot(self.dates, np.exp(self.sizes))
        ax.set_xlabel("Date")
        ax.set_ylabel("Size")
        ax.set_title(f"Lesion Trajectory for Patient {self.patient_id}, Lesion {self.label_id}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1 * np.max(np.exp(self.sizes)))
        formatter = FuncFormatter(lambda x, p: f'{x:.2e}')
        ax.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.show()

    def plot_animation(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((500, 500)), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Lesion Trajectory for Patient {self.patient_id}, Lesion {self.label_id}")

        def update(frame):
            data_segmentation, _ = self.load_labels_for_inr(selected_date=self.dates[frame])
            data_2d = np.max(data_segmentation, axis=2)
            im.set_data(data_2d)

        ani = animation.FuncAnimation(fig, update, frames=len(self.dates), repeat=False)
        plt.close()
        return HTML(ani.to_jshtml())

    def extract_growth_phase(self, sizes, smoothing_window=5, min_start_idx=1):
        """
        Extract the main growth phase of a lesion trajectory, ignoring early peaks.

        Args:
            sizes: array of lesion sizes (can be log-scale)
            smoothing_window: window size for Savitzky-Golay filter
            min_start_idx: minimum index to consider as peak (avoids very early spikes)

        Returns:
            start_idx, end_idx, plot_data
        """
        sizes_array = np.array(sizes)
        n = len(sizes_array)

        if n > smoothing_window:
            smoothed = savgol_filter(sizes_array, smoothing_window, 2)
        else:
            smoothed = sizes_array

        derivative = np.gradient(smoothed)

        valid_range_start = min(min_start_idx, n - 2)
        peak_idx = valid_range_start + np.argmax(smoothed[valid_range_start:])

        threshold = np.std(derivative) * 0.5

        growth_start_candidates = np.where(derivative[:peak_idx] > threshold)[0]
        if len(growth_start_candidates) > 0:
            start_idx = growth_start_candidates[0]
        else:
            start_idx = max(0, peak_idx - 3)

        growth_end_candidates = np.where(derivative[peak_idx:] < -threshold)[0]
        if len(growth_end_candidates) > 0:
            end_idx = peak_idx + growth_end_candidates[0]
        else:
            end_idx = min(n - 1, peak_idx + 1)

        start_idx = max(0, min(start_idx, n - 1))
        search_start_left = max(0, start_idx - 5)
        search_start_right = min(n - 1, start_idx + 5)

        best_start = start_idx
        best_start_value = sizes_array[start_idx]

        for idx in range(search_start_left, search_start_right + 1):
            if sizes_array[idx] < best_start_value:
                best_start = idx
                best_start_value = sizes_array[idx]

        start_idx = best_start

        end_idx = max(0, min(end_idx, n - 1))
        search_end_left = max(0, end_idx - 5)
        search_end_right = min(n - 1, end_idx + 5)

        best_end = end_idx
        best_end_value = sizes_array[end_idx]

        for idx in range(search_end_left, search_end_right + 1):
            if sizes_array[idx] > best_end_value:
                best_end = idx
                best_end_value = sizes_array[idx]

        end_idx = best_end

        if start_idx >= end_idx:
            start_idx = max(0, end_idx - 1)

        for idx in range(start_idx - 1, max(-1, start_idx - 10), -1):
            if idx >= 0 and idx < end_idx:
                if sizes_array[idx] <= sizes_array[start_idx] + np.log(1.05):
                    start_idx = idx
                else:
                    break

        for idx in range(end_idx + 1, min(n, end_idx + 10)):
            if idx > start_idx:
                if sizes_array[idx] >= sizes_array[end_idx] + np.log(0.95):
                    end_idx = idx
                else:
                    break

        return start_idx, end_idx, {'smoothed': smoothed, 'derivative': derivative, 'peak': peak_idx}