"""
Patient ties together all scans for one patient, registration, lesion
tracking across time, and trajectory building/merging.

Two things changed from the original monolithic version, on top of the
module split:

1. `dataloader` and `registrator` used to default to `MRI_Dataloader()` /
   `Registrator()` evaluated once at *function-definition* time (a classic
   Python mutable-default-argument bug). That meant every Patient created
   without explicit args shared the exact same dataloader instance, and
   simply importing this module would eagerly glob the entire dataset.
   Both now default to `None` and are lazily constructed inside `__init__`.

2. `MRI_Dataloader` and `Lesion_Trajectory` are imported lazily (inside
   `__init__` / the methods that use them) because those modules import
   `Patient` back -- a genuine mutual dependency, not just a shared path
   string. `Registrator` has no such cycle, so it's imported normally at
   module level.
"""

import glob
import json
import itertools

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours, marching_cubes
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

from .registrator import Registrator


class Patient:
    def __init__(self, patient_id, dataloader=None, registrator=None):
        self.patient_id = patient_id

        if dataloader is None:
            from .mri_dataloader import MRI_Dataloader  # lazy: see module docstring
            dataloader = MRI_Dataloader()
        if registrator is None:
            registrator = Registrator()

        self.dataloader = dataloader
        self.samples = self.dataloader.find_by_patient_id(patient_id)
        self.dates = [sample.date for sample in self.samples]
        self.path = self.dataloader.paths.sample_dir(patient_id)
        self.registrator = registrator
        self.load_registered_transforms()
        self.patient_trajectory_paths = glob.glob(self.dataloader.paths.lesion_trajectory_glob(patient_id))

    def register_all_to_first(self, registrator: Registrator = None):
        """Register all images to the first image using linear translation only."""
        registrator = registrator or Registrator()
        reference_image = self.samples[0].load_mri()

        self.samples[0].registered_transform = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0])
        self.samples[0].save_registered_images()

        registered_transforms = []

        for sample in self.samples[1:]:
            moving_image = sample.load_mri()

            transformation = registrator.register(reference_image, moving_image)
            sample.registered_transform = transformation
            sample.save_registered_images()

            registered_transforms.append(transformation)
            print(f"Registered {sample.date} to {self.samples[0].date} with transformation:\n{transformation}")

        return registered_transforms

    def plot_3d_lesion_position(self, log_size=True):
        from sklearn.cluster import KMeans

        D_point, D_sizes, D_time, D_time_absolute = [], [], [], []
        data = {}

        for i, mri in enumerate(self.samples):
            mri.load_mri_segmentation()

            data[mri.date] = {
                "num_features": mri.num_lesions,
                "lesion_sizes": mri.lesion_sizes,
                "lesion_coords": mri.lesion_coords,
            }

            for n in range(mri.lesion_coords.shape[0]):
                D_point.append(mri.lesion_coords[n])
                D_sizes.append(np.log(mri.lesion_sizes[n]))
                D_time.append(i)
                D_time_absolute.append(mri.date)

        maximum_n = max([len(data[d]["lesion_sizes"]) for d in data.keys()])

        D_point = np.stack(D_point)

        kmeans = KMeans(n_clusters=maximum_n, random_state=0, n_init=200).fit(D_point)
        labels = kmeans.labels_

        df = pd.DataFrame({
            "x": D_point[:, 0],
            "y": D_point[:, 1],
            "z": D_point[:, 2],
            "log_size": D_sizes,
            "size": np.exp(D_sizes),
            "time": D_time,
            "time_absolute": D_time_absolute,
            "labels": labels,
            "patient": self.patient_id
        })
        scale = 'log_size' if log_size else 'size'
        fig = px.scatter_3d(df, x='x', y='y', z='z', size=scale, color='time', height=800, width=900, symbol=labels)

        fig.show()

    def contur_plot(self, fig, data, col):
        data = data > 0.0

        contours_3d = []

        for z in range(data.shape[-1]):
            slice_2d = data[:, :, z]

            contours = find_contours(slice_2d, level=0.5)

            for contour in contours:
                x = contour[:, 0]
                y = contour[:, 1]
                z_coords = np.full_like(x, z)

                contours_3d.append((x, y, z_coords))

        for x, y, z in contours_3d:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=2, color=col), opacity=0.6))

    def plot_image_registration(self, registered_parameters=None, registrator: Registrator = None):
        registered_parameters = registered_parameters or []
        registrator = registrator or Registrator()

        colors = (qualitative.Dark24)
        color_cycle = itertools.cycle(colors)

        fig = go.Figure()

        for i, sample in enumerate(self.samples):
            col = next(color_cycle)

            if hasattr(sample, "registered_transform") and sample.registered_transform is not None:
                rp = sample.registered_transform
            elif registered_parameters:
                rp = registered_parameters[i]
            else:
                rp = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0])
                print(f"Sample {i} has no registered transform. Using no transformation.")

            points_transformed = registrator.rigid_transform(
                sample.load_mri(),
                rp,
                self.samples[0].load_mri().shape
            )

            self.contur_plot(fig, points_transformed, col)

        fig.update_layout(
            height=800,
            width=900
        )

        fig.show()

    def plot_registered_3d_lesion_position(self, registered_parameters=None, log_size=True, relative_size=False):
        """
        registered_parameters: List of parameter vectors (length 9) for each sample.
                registered_parameters[0] should be the identity: [0,0,0,0,0,0,1,1,1]
        """
        from sklearn.cluster import AffinityPropagation
        registered_parameters = registered_parameters or []

        D_point, D_sizes, D_relative_sizes, D_time, D_time_absolute = [], [], [], [], []

        ref_shape = self.samples[0].load_mri().shape
        for i, mri in enumerate(self.samples):

            mri.load_mri_segmentation()
            if not hasattr(mri, 'lesion_coords'):
                mri.process_sample()

            if hasattr(mri, "registered_transform") and mri.registered_transform is not None:
                params = mri.registered_transform
            elif registered_parameters:
                params = registered_parameters[i]
            else:
                params = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0])
                print(f"Sample {i} has no registered transform. Using no transformation.")

            for n in range(len(mri.lesion_coords)):
                moving_point = np.array(mri.lesion_coords[n])

                fixed_point = moving_point

                if i > 0:
                    tx, ty, tz, _, _, _, sx, sy, sz = params
                    fixed_point = (moving_point - np.array([tx, ty, tz])) / np.array([sx, sy, sz])

                D_point.append(fixed_point)
                D_sizes.append(mri.lesion_sizes[n])
                D_relative_sizes.append(mri.relative_lesion_sizes[n])
                D_time.append(i)
                D_time_absolute.append(mri.date)

        if not D_point:
            print("No lesions found.")
            return

        D_point = np.stack(D_point)
        D_relative_sizes = (D_relative_sizes - min(D_relative_sizes)) / (max(D_relative_sizes) - min(D_relative_sizes) + 1e-8) * 10

        affprop = AffinityPropagation(random_state=0).fit(D_point)

        df = pd.DataFrame({
            "x": D_point[:, 1],
            "y": ref_shape[0] - D_point[:, 0],
            "z": D_point[:, 2],
            "size": D_sizes,
            "relative_size": D_relative_sizes,
            "log_size": np.log(D_sizes),
            "time": D_time,
            "date": D_time_absolute,
            "lesion_id": affprop.labels_.astype(str)
        })

        scale_col = 'log_size' if log_size else 'size'
        if relative_size:
            scale_col = 'relative_size'

        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            size=scale_col,
            size_max=30,
            color='time',
            symbol='lesion_id',
            hover_data=['date', 'size'],
            title=f"Longitudinal Lesion Tracking: Patient {self.patient_id}",
            height=800, width=1000
        )

        for x, y, z in self.samples[0].calculate_contours():
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=1, color="rgba(50,50,50,1)"),
                showlegend=False
            )
            )

        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=300 / ref_shape[0], y=300 / ref_shape[1], z=1)
            )
        )

        fig.show()

    def __repr__(self):
        return f"Patient {self.patient_id}, {len(self.samples)} scans"

    def plot_average_slice_trajectory(self):

        means = []
        for i, sample in enumerate(self.samples):
            means.append(sample.load_mri().mean(axis=(0, 1)))

        means_interp = []
        for mean in means:
            m = mean.copy()

            sliced_mean = m[m != 0]

            x_old = np.linspace(0, 1, len(sliced_mean))
            x_new = np.linspace(0, 1, 100)
            mean_interp = np.interp(x_new, x_old, sliced_mean)
            mean_interp = mean_interp / np.sort(mean_interp)[-5]
            means_interp.append(mean_interp)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        cmap = plt.get_cmap("cool")
        colors = cmap(np.linspace(0, 1, len(means)))
        for i, mean in enumerate(means):
            axes[0].plot(mean, label=f"Sample {i}", color=colors[i])
        axes[0].set_title("Mean MRI Intensity per Slice")
        axes[0].set_xlabel("Slice Index")
        axes[0].set_ylabel("Mean Intensity")
        axes[0].legend()

        for i, mean_interp in enumerate(means_interp):
            axes[1].plot(mean_interp, label=f"Sample {i} (Interpolated)", color=colors[i])
        axes[1].set_title("Interpolated Mean Intensity (100 pts)")
        axes[1].set_xlabel("Interpolated Index")
        axes[1].set_ylabel("Normalized Intensity")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def process_samples(self):
        for sample in self.samples:
            sample.process_sample()

    def load_registered_transforms(self):
        self.registered_transforms = []
        for sample in self.samples:
            if hasattr(sample, "registered_transform") and sample.registered_transform is not None:
                self.registered_transforms.append(sample.registered_transform)
            else:
                self.registered_transforms.append(sample.load_registered_transform())
        return self.registered_transforms

    def _transform_points(self, points, transform_params):
        """
        Apply rigid transformation to 3D points.

        Args:
            points: Nx3 array of 3D points
            transform_params: 9-element array [tx, ty, tz, rx, ry, rz, sx, sy, sz]

        Returns:
            Transformed Nx3 array
        """
        tx, ty, tz, rx, ry, rz, sx, sy, sz = transform_params

        transformed = points.copy()
        transformed[:, 0] = (points[:, 0] - tx) / sx
        transformed[:, 1] = (points[:, 1] - ty) / sy
        transformed[:, 2] = (points[:, 2] - tz) / sz

        return transformed

    def _match_lesions_between_scans(self, scan_idx, prev_centroids, prev_sizes, curr_centroids,
                                      curr_sizes, distance_threshold=None):
        """
        Find optimal matching between lesions in CONSECUTIVE scans using Hungarian algorithm.

        **Simplified approach without registration transforms:**
        - Matches based on spatial proximity in original scan coordinates
        - Uses generous distance thresholds (brain doesn't move much between scans)
        - Validates with size ratios

        Args:
            scan_idx: Current scan index (only for debugging)
            prev_centroids: Mx3 array of centroids from previous scan
            prev_sizes: M array of lesion sizes from previous scan
            curr_centroids: Nx3 array of centroids from current scan
            curr_sizes: N array of lesion sizes from current scan
            distance_threshold: Maximum distance for matching. If None, uses 100 pixels
            debug: If True, print diagnostic information

        Returns:
            List of (prev_idx, curr_idx) tuples for valid matches
        """
        if len(prev_centroids) == 0 or len(curr_centroids) == 0:
            return []

        prev_centroids_z_corrected = prev_centroids * [1,1,10]
        curr_centroids_z_corrected = curr_centroids * [1,1,10]


        if distance_threshold is None:
            distance_threshold = 100

        distances = np.linalg.norm(
            prev_centroids_z_corrected[:, np.newaxis, :] - curr_centroids_z_corrected[np.newaxis, :, :],
            axis=2
        )

        size_ratio_matrix = np.zeros_like(distances)
        for i in range(len(prev_centroids_z_corrected)):
            for j in range(len(curr_centroids_z_corrected)):
                ratio = curr_sizes[j] / (prev_sizes[i] + 1e-6)
                if ratio < 0.3 or ratio > 3.0:
                    size_ratio_matrix[i, j] = 100
                elif ratio < 0.5 or ratio > 2.0:
                    size_ratio_matrix[i, j] = 10

        cost_matrix = distances + size_ratio_matrix

        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for p_idx, c_idx in zip(prev_indices, curr_indices):
            distance = distances[p_idx, c_idx]
            size_ratio = curr_sizes[c_idx] / (prev_sizes[p_idx] + 1e-6)

            if distance < distance_threshold and 0.2 <= size_ratio <= 5.0:
                matches.append((p_idx, c_idx))

        return matches

    def _find_continuation(self, scan_idx, lesion_centroids, lesion_sizes,
                            scan_lesions, visited, max_gap=2, spatial_threshold=50,
                            allow_size_change=2.0):
        """
        Find the next occurrence of a lesion, allowing gaps.

        Args:
            scan_idx: Current scan index
            lesion_centroids: Centroid of current lesion
            lesion_sizes: Size of current lesion
            scan_lesions: List of all scan lesion data
            visited: Set of already-visited (scan, lesion_idx) tuples
            max_gap: Maximum number of scans to look ahead
            spatial_threshold: Max distance to search
            allow_size_change: Max size ratio to allow (e.g., 2.0 = up to 2x size change)

        Returns:
            (next_scan, next_lesion_idx) or (None, None) if not found
        """
        for future_scan in range(scan_idx + 1, min(scan_idx + 1 + max_gap, len(scan_lesions))):
            if len(scan_lesions[future_scan]['centroids']) == 0:
                continue

            best_dist = float('inf')
            best_idx = None

            for lesion_idx in range(len(scan_lesions[future_scan]['labels'])):
                if (future_scan, lesion_idx) in visited:
                    continue

                future_centroid = scan_lesions[future_scan]['centroids'][lesion_idx]
                future_size = scan_lesions[future_scan]['sizes'][lesion_idx]

                dist = np.linalg.norm(lesion_centroids - future_centroid)
                size_ratio = future_size / (lesion_sizes + 1e-6)

                inv_ratio = (lesion_sizes + 1e-6) / future_size if future_size > 0 else float('inf')
                size_valid = (1.0 / allow_size_change <= size_ratio <= allow_size_change)

                if dist < spatial_threshold and size_valid:
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = lesion_idx

            if best_idx is not None:
                return future_scan, best_idx

        return None, None

    def _build_trajectories_from_matches(self, scan_lesions, matches, num_scans,
                                          max_gap=2, spatial_threshold=50,
                                          size_ratio_threshold=4.0):
        """
        Build lesion trajectories by following match chains across scans, allowing gaps.

        Key feature: Seeds trajectories from ALL scans, not just the first one.
        This ensures that:
        - Lesions present in scan 0 start trajectories forward
        - NEW lesions appearing in scan i (unmatched) start fresh trajectories
        - Every detected lesion is part of exactly one trajectory

        Trajectories can skip timepoints if a lesion is absent (labeling error, registration
        issue, or real disappearance). Only scans where the lesion is actually present are
        included in the trajectory.

        Args:
            scan_lesions: List of dicts with 'centroids', 'sizes', 'labels' for each scan
            matches: List of (scan_idx, prev_idx, curr_idx) tuples
            num_scans: Number of scans
            max_gap: Max scans to skip (default 2)
            spatial_threshold: Max distance for gap-filling (default 50)
            size_ratio_threshold: Allow size changes up to this factor (default 4.0)

        Returns:
            List of trajectory dicts: {scan_indices: [...], labels: [(scan_idx, label), ...], sizes: [...]}
        """
        adj = {}
        for scan_idx, prev_idx, curr_idx in matches:
            key = (scan_idx, prev_idx)
            adj[key] = (scan_idx + 1, curr_idx)

        visited = set()
        trajectories = []

        for lesion_idx in range(len(scan_lesions[0]['labels'])):
            if (0, lesion_idx) in visited:
                continue

            trajectory = self._build_single_trajectory(
                scan_idx=0, lesion_idx=lesion_idx,
                scan_lesions=scan_lesions, adj=adj, visited=visited,
                num_scans=num_scans, max_gap=max_gap,
                spatial_threshold=spatial_threshold,
                size_ratio_threshold=size_ratio_threshold
            )
            trajectories.append(trajectory)

        for scan_idx in range(1, num_scans):
            for lesion_idx in range(len(scan_lesions[scan_idx]['labels'])):
                if (scan_idx, lesion_idx) in visited:
                    continue

                trajectory = self._build_single_trajectory(
                    scan_idx=scan_idx, lesion_idx=lesion_idx,
                    scan_lesions=scan_lesions, adj=adj, visited=visited,
                    num_scans=num_scans, max_gap=max_gap,
                    spatial_threshold=spatial_threshold,
                    size_ratio_threshold=size_ratio_threshold
                )
                trajectories.append(trajectory)

        return trajectories

    def _build_single_trajectory(self, scan_idx, lesion_idx, scan_lesions, adj, visited,
                                  num_scans, max_gap, spatial_threshold, size_ratio_threshold):
        """
        Build a single trajectory starting from (scan_idx, lesion_idx).
        Follows matches forward and handles gaps.
        """
        trajectory = {
            'scan_indices': [scan_idx],
            'labels': [(scan_idx, scan_lesions[scan_idx]['labels'][lesion_idx])],
            'sizes': [scan_lesions[scan_idx]['sizes'][lesion_idx]],
            'centroids': [scan_lesions[scan_idx]['centroids'][lesion_idx]]
        }
        visited.add((scan_idx, lesion_idx))

        current_scan = scan_idx
        current_lesion_idx = lesion_idx
        current_centroid = scan_lesions[scan_idx]['centroids'][lesion_idx]
        current_size = scan_lesions[scan_idx]['sizes'][lesion_idx]

        while current_scan < num_scans - 1:
            next_scan = None
            next_lesion_idx = None

            # 1. Try direct adjacency match first
            if (current_scan, current_lesion_idx) in adj:
                adj_scan, adj_lesion_idx = adj[(current_scan, current_lesion_idx)]
                
                # CRITICAL FIX: Only accept the direct match if it hasn't been claimed
                # by a gap-fill from another trajectory!
                if (adj_scan, adj_lesion_idx) not in visited:
                    next_scan = adj_scan
                    next_lesion_idx = adj_lesion_idx

            # 2. If no valid direct match is available, try gap-filling
            if next_scan is None:
                next_scan, next_lesion_idx = self._find_continuation(
                    current_scan, current_centroid, current_size,
                    scan_lesions, visited, max_gap=max_gap,
                    spatial_threshold=spatial_threshold,
                    allow_size_change=size_ratio_threshold
                )

            # 3. If a valid next step was found (via either method), update state
            if next_scan is not None:
                trajectory['scan_indices'].append(next_scan)
                trajectory['labels'].append((next_scan, scan_lesions[next_scan]['labels'][next_lesion_idx]))
                trajectory['sizes'].append(scan_lesions[next_scan]['sizes'][next_lesion_idx])
                trajectory['centroids'].append(scan_lesions[next_scan]['centroids'][next_lesion_idx])

                visited.add((next_scan, next_lesion_idx))
                current_scan = next_scan
                current_lesion_idx = next_lesion_idx
                current_centroid = scan_lesions[next_scan]['centroids'][next_lesion_idx]
                current_size = scan_lesions[next_scan]['sizes'][next_lesion_idx]
            else:
                # No valid direct match and no continuation found, end the trajectory
                break

        return trajectory

    def merge_lesion_to_trajectory(self, max_gap=7, spatial_threshold=40,
                                    size_ratio_threshold=12.0, distance_threshold=40):
        """
        Match lesions across scans using lenient spatial matching.
        Uses Hungarian algorithm for optimal 1-to-1 correspondence between consecutive scans.

        **Approach**: Direct spatial matching without registration transforms.
        - Matches based on centroid proximity (brain position relatively stable across consecutive scans)
        - Allows generous spatial thresholds
        - Implements smart gap-filling for missing lesions at timepoints

        Handles realistic scenario where:
        - Some lesions disappear (merge, resolve, or labeling errors)
        - New lesions appear
        - Variable number of lesions per timepoint

        Args:
            max_gap: Max scans to skip when looking for lesion continuation (default 2)
            spatial_threshold: Max pixel distance for gap-filling matches (default 150)
            size_ratio_threshold: Allow size changes up to this factor (default 3.0 = 3x)
            distance_threshold: Max distance for consecutive scan matches (default 100)
        """
        from .lesion_trajectory import Lesion_Trajectory 

        vol_shape = (500, 500, 50)
        volume, affine = self.samples[0].load_mri(zoomed=True, affine=True)
        volume = np.sum(volume[volume > 0])

        scan_lesions = []
        lesion_counts = []
        for i, sample in enumerate(self.samples): # first step: extract all the data from the mri scans
            labeled_mask = sample.load_mri_segmentation(zoomed=True)
            num_lesions = int(labeled_mask.max())
            lesion_counts.append(num_lesions)

            if num_lesions > 0:
                centroids = ndimage.center_of_mass(
                    labeled_mask,
                    labeled_mask,
                    range(1, num_lesions + 1)
                )
                centroids = np.array(centroids)

                sizes = ndimage.sum(
                    np.ones_like(labeled_mask),
                    labeled_mask,
                    range(1, num_lesions + 1)
                )
            else:
                centroids = np.empty((0, 3))
                sizes = np.array([])

            scan_lesions.append({
                'centroids': centroids,
                'sizes': sizes,
                'labels': np.arange(1, num_lesions + 1),
                'labeled_mask': labeled_mask
            })
            print(f"  Scan {i} ({self.samples[i].date}): {num_lesions} lesions")


        matches = []
        unmatched_by_scan = []

        for i in range(1, len(self.samples)): # match the scans
            prev_centroids = scan_lesions[i - 1]['centroids']
            prev_sizes = scan_lesions[i - 1]['sizes']
            curr_centroids = scan_lesions[i]['centroids']
            curr_sizes = scan_lesions[i]['sizes']


            scan_matches = self._match_lesions_between_scans(
                i, prev_centroids, prev_sizes, curr_centroids, curr_sizes,
                distance_threshold=distance_threshold
            )

            matched_prev = set(m[0] for m in scan_matches)
            unmatched_count = len(prev_centroids) - len(matched_prev)

            for prev_idx, curr_idx in scan_matches:
                matches.append((i - 1, prev_idx, curr_idx))

            unmatched_by_scan.append((i, unmatched_count))

        trajectories = self._build_trajectories_from_matches(
            scan_lesions, matches, len(self.samples),
            max_gap=max_gap, spatial_threshold=spatial_threshold,
            size_ratio_threshold=size_ratio_threshold
        )

        trajectories_with_gaps = sum(1 for t in trajectories if len(t['scan_indices']) < len(self.samples))
        print(f"Found {len(trajectories)} lesion trajectories")
        print(f"  - {trajectories_with_gaps} trajectories have gaps (missing timepoints)")
        print(f"  - {len(trajectories) - trajectories_with_gaps} trajectories span all timepoints")
        trajectories_filtered = [t for t in trajectories if len(t['scan_indices']) > 2]
        trajectories_short = len(trajectories) - len(trajectories_filtered)

        if trajectories_short > 0:
            print(f"\nFiltering: Removing {trajectories_short} trajectories with ≤2 entries")
            print(f"Keeping {len(trajectories_filtered)} trajectories with >2 entries")

        trajectories = trajectories_filtered

        labeled_mask_cache = [np.zeros(vol_shape, dtype=np.uint8) for _ in range(len(self.samples))]
        trajectory_sizes = np.zeros((len(self.samples), len(trajectories)), dtype=np.uint32)

        for traj_id, trajectory in enumerate(trajectories):
            sample_ids = trajectory['scan_indices']

            valid_frames = []
            removed_frames = []

            for frame_idx, scan_idx in enumerate(sample_ids):
                orig_mask = scan_lesions[scan_idx]['labeled_mask']
                orig_label = trajectory['labels'][frame_idx][1]
                lesion_mask = (orig_mask == orig_label).astype(np.uint8)
                voxel_count = np.sum(lesion_mask)

                if voxel_count > 0:
                    valid_frames.append(frame_idx)
                else:
                    removed_frames.append((scan_idx, self.samples[scan_idx].date, voxel_count))

            if removed_frames:
                print(f"  Trajectory {traj_id + 1}: Removing {len(removed_frames)} empty frames")
                for scan_idx, date, voxels in removed_frames[:3]:
                    print(f"    - Scan {scan_idx} ({date}): {voxels} voxels")
                if len(removed_frames) > 3:
                    print(f"    ... and {len(removed_frames) - 3} more")

            if len(valid_frames) == 0:
                print(f"  Trajectory {traj_id + 1}: SKIPPED - all {len(sample_ids)} frames are empty!")
                continue

            if len(valid_frames) < len(sample_ids):
                cleaned_trajectory = {
                    'scan_indices': [sample_ids[i] for i in valid_frames],
                    'labels': [trajectory['labels'][i] for i in valid_frames],
                    'sizes': [trajectory['sizes'][i] for i in valid_frames],
                    'centroids': [trajectory['centroids'][i] for i in valid_frames]
                }
                sample_ids_to_save = cleaned_trajectory['scan_indices']
                print(f"    Final: {len(valid_frames)}/{len(trajectory['scan_indices'])} valid frames")
            else:
                cleaned_trajectory = trajectory
                sample_ids_to_save = sample_ids

            for scan_idx in sample_ids_to_save: # scan_idx is compareable to the time or the date
                # we now want to get the label, how it was in the nibabel label file of the individual classification 
                orig_label = cleaned_trajectory['labels'][list(zip(*cleaned_trajectory['labels']))[0].index(scan_idx)][1]

                # then we get from the list that saves all of the time points we want to export the actual labeled mask, but we still have all the other labels like 0,1,2,3, ... in them
                orig_mask = scan_lesions[scan_idx]['labeled_mask']

                # then we get a mask from only the labels that we actually want to have in that round
                lesion_mask = (orig_mask == orig_label).astype(np.uint8)

                # we set all the coordiantes to the global trajectory index
                assert all(labeled_mask_cache[scan_idx][lesion_mask > 0] == 0)   
                labeled_mask_cache[scan_idx][lesion_mask > 0] = traj_id + 1

                # we export the size of it
                trajectory_sizes[scan_idx, traj_id] = np.sum(lesion_mask)

            log_sizes = np.log(cleaned_trajectory['sizes'] / (volume + 1e-7) + 1e-7)
            trajectory_obj = Lesion_Trajectory(
                patient_id=self.patient_id,
                label_id=traj_id + 1,
                sample_ids=sample_ids_to_save,
                sizes=log_sizes
            )
            trajectory_obj.save_lesion_trajectory()

        print("\nSaving relabeled masks...")
        for i, sample in enumerate(self.samples):
            img = nib.Nifti1Image(labeled_mask_cache[i], affine)
            metadata = {
                "num_features": len(trajectories),
                "sizes": np.log(trajectory_sizes[i] / (volume + 1e-7) + 1e-7).tolist()
            }

            json_str = json.dumps(metadata)
            extension = nib.nifti1.Nifti1Extension(44, json_str.encode('utf-8'))
            img.header.extensions.append(extension)
            nib.save(img, sample.lesion_trajectory_path)

        print("Trajectory building complete!")

    def diagnose_trajectory_empty_frames(self, trajectory_idx=1):
        """
        Diagnose if a trajectory has empty frames (no actual lesion data at some timepoints).
        Helpful for debugging gap-filled trajectories.

        Args:
            trajectory_idx: Which trajectory to examine (1-indexed from saved trajectories)
        """
        trajectories = self.load_lesion_trajectories()
        if trajectory_idx < 1 or trajectory_idx > len(trajectories):
            print(f"Invalid trajectory index. Available: 1-{len(trajectories)}")
            return

        trj = trajectories[trajectory_idx - 1]
        print(f"\n=== Diagnosing Trajectory {trajectory_idx} (Label {trj.label_id}) ===")
        print(f"Patient: {trj.patient_id}")
        print(f"Number of scans in trajectory: {len(trj.dates)}")
        print(f"Dates: {trj.dates}")
        print(f"Sizes (log): {trj.sizes}")

        print("\nLoading trajectory data for all dates:")
        data_all = trj.load_labels_for_inr()

        for i, (data_seg, time_point) in enumerate(data_all):
            non_zero_voxels = np.count_nonzero(data_seg)
            print(f"  {trj.dates[i]}: {non_zero_voxels} voxels (time_point={time_point:.3f})")

            if non_zero_voxels == 0:
                print(f"    ^ WARNING: Frame is EMPTY!")

        return data_all

    def load_lesion_trajectories(self):
        from .lesion_trajectory import Lesion_Trajectory  # lazy: see module docstring
        trajectories = []
        for path in self.patient_trajectory_paths:
            trajectories.append(Lesion_Trajectory(load_from_trajectory_path=path))
        return trajectories

    def plot_lesion_shape_trajectory(self):
        fig = go.Figure()

        colors = px.colors.sequential.Agsunset

        vol_shape = self.samples[0].load_mri().shape

        for i, sample in enumerate(self.samples):
            labeled_mask = sample.load_mri_segmentation()
            params = self.registered_transforms[i]
            labeled_mask = self.registrator.rigid_transform(labeled_mask, params, vol_shape)

            color_idx = int((i / len(self.samples)) * (len(colors) - 1))
            time_color = colors[color_idx]

            unique_labels = np.unique(labeled_mask)
            for label in unique_labels:
                if label == 0:
                    continue

                lesion_mask = (labeled_mask == label).astype(np.uint8)

                try:
                    verts, faces, _, _ = marching_cubes(lesion_mask, step_size=2, allow_degenerate=True, method="lewiner")

                    n_samples = 10
                    sampled_points = []

                    for _ in range(n_samples):
                        face_idx = np.random.randint(0, len(faces))
                        face = faces[face_idx]

                        r1, r2 = np.random.random(2)
                        if r1 + r2 > 1:
                            r1 = 1 - r1
                            r2 = 1 - r2

                        point = (1 - r1 - r2) * verts[face[0]] + r1 * verts[face[1]] + r2 * verts[face[2]]
                        sampled_points.append(point)

                    sampled_points = np.array(sampled_points)

                    fig.add_trace(go.Scatter3d(
                        x=sampled_points[:, 1],
                        y=vol_shape[0] - sampled_points[:, 0],
                        z=sampled_points[:, 2],
                        mode="markers",
                        marker=dict(size=5, color=time_color, opacity=0.3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                except RuntimeError:
                    continue

        for x, y, z in self.samples[0].calculate_contours():
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=1, color="rgba(50,50,50,1)"),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=f"3D Lesion Shape Trajectory for Patient {self.patient_id}",
            width=1000,
            height=800
        )

        fig.show()