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

            from .patient import Patient  # lazy: genuine mutual dependency, see module docstring
            patient = Patient(patient_id)
            for i, sample in enumerate(patient.samples):
                if i not in sample_ids:
                    continue
                sample.load_mri_segmentation()
                self.dates.append(sample.date)
            self.n_scans = len(sample_ids)
            self.sizes = sizes
            self.extract_growth_phase()

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
            self.extract_growth_phase()
        else:
            print(f"Lesion trajectory file not found at {self.path}. Cannot load trajectory.")

    def load_sizes(self):
        sizes = []

        from .patient import Patient  # lazy: see module docstring
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
                data_segmentation = np.where(np.isin(data_segmentation, [self.label_id]), 1, 0).astype(np.uint8)

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

    def extract_growth_phase(self, sizes=None, tolerance=0.15, max_skip=6, log_scale=True):
        """
        Extract the main growth phase of a lesion trajectory using a walker-based
        approach: for every candidate starting index, spawn a backward walker
        (checks the lesion got smaller going into the past) and a forward walker
        (checks the lesion got bigger going into the future). Each walker tolerates
        up to `tolerance` fractional violation of the trend, and if it gets stuck,
        tries skipping ahead up to `max_skip` scans to find a continuation.

        The longest resulting trajectory (by number of accepted sample indices,
        across all starting points) is returned.

        Args:
            sizes: array of lesion sizes (linear scale by default; see log_scale)
            tolerance: fractional tolerance (e.g. 0.15 = 15%) for a step to still
                count as "still growing" / "still shrinking"
            max_skip: max number of scans a walker may skip over when stuck,
                before trying to look further, to find a valid continuation
            log_scale: if True, sizes are assumed to already be log-transformed,
                and the tolerance is applied additively as log(1 + tolerance)
                instead of multiplicatively

        Returns:
            start_idx, end_idx, plot_data
        """
        if sizes is None:
            sizes = self.sizes

        sizes_array = np.array(sizes, dtype=float)
        n = len(sizes_array)

        if n == 0:
            return 0, 0, {'segments': [], 'best_segment': None}

        if log_scale:
            tol_up = np.log1p(tolerance)     # additive tolerance in log-space
            tol_down = np.log1p(tolerance)
        else:
            tol_up = tolerance
            tol_down = tolerance

        def is_valid_step(extreme_value, candidate_value, direction):
            if direction == -1:
                # backward walker: candidate should be within tolerance of the lowest point seen so far
                if log_scale:
                    return candidate_value <= extreme_value + tol_down
                else:
                    return candidate_value <= extreme_value * (1 + tol_down)
            else:
                # forward walker: candidate should be within tolerance of the highest point seen so far
                if log_scale:
                    return candidate_value >= extreme_value - tol_up
                else:
                    return candidate_value >= extreme_value * (1 - tol_up)

        def walk(start_idx, direction):
            """Walks in one direction from start_idx, returning the list of
            accepted indices (including start_idx), in the order visited.

            Tolerance is measured against the running extremum encountered so far
            along the walk (minimum for the backward walker, maximum for the
            forward walker) rather than against the immediately preceding point.
            """
            trajectory = [start_idx]
            idx = start_idx
            extreme_value = sizes_array[start_idx]  # running min (direction=-1) or max (direction=+1)

            while True:
                found_continuation = False
                for skip in range(0, max_skip + 1):
                    candidate_idx = idx + direction * (1 + skip)
                    if candidate_idx < 0 or candidate_idx >= n:
                        break
                    candidate_value = sizes_array[candidate_idx]
                    if is_valid_step(extreme_value, candidate_value, direction):
                        idx = candidate_idx
                        trajectory.append(candidate_idx)
                        found_continuation = True
                        # update the running extremum
                        if direction == -1:
                            extreme_value = min(extreme_value, candidate_value)
                        else:
                            extreme_value = max(extreme_value, candidate_value)
                        break
                if not found_continuation:
                    break

            return trajectory

        all_segments = []
        for start in range(n):
            back_traj = walk(start, direction=-1)
            fwd_traj = walk(start, direction=+1)

            full_indices = sorted(set(back_traj) | set(fwd_traj))
            all_segments.append({
                'origin': start,
                'indices': full_indices,
                'length': len(full_indices),
            })

        best_segment = max(all_segments, key=lambda s: s['length'])
        start_idx = min(best_segment['indices'])
        end_idx = max(best_segment['indices'])

        self.allowed_dates = [self.dates[i] for i in best_segment['indices']]

        return start_idx, end_idx, best_segment['indices']

    def plot_growth_phase(self):
        data = self.extract_growth_phase(log_scale=True, tolerance=0.15)
        plt.figure(figsize=(12, 6))
        plt.plot(self.sizes, label='Original Sizes')
        plt.axvline(data[0], color='g', linestyle='--', label='Growth Start')
        plt.axvline(data[1], color='r', linestyle='--', label='Growth End')
        plt.scatter(data[2], self.sizes[data[2]], label='Final_samples')
        plt.legend()
        plt.title(f'Lesion Size Trajectory with Growth Phase of {self.patient_id} Lesion {self.label_id}')
        plt.xlabel('Scan Index')
        plt.ylabel('Lesion Size (log-scale)')
        plt.tight_layout()
        plt.show()