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

import os
import io
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image as PILImage
from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import find_contours, marching_cubes
import plotly.express as px
import plotly.graph_objects as go

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

    def register_all_to_first(self):
        """Register all images to the first image using linear translation only."""
        import ants

        ants.set_num_threads(2)

        fixed_image = ants.image_read(self.samples[0].original_sample_path)

        for sample in self.samples:
            if not os.path.exists(sample.original_sample_path) or not os.path.exists(sample.lesion_segmentation_path):
                print(f"Skipping registration for {sample.patient_id} on {sample.date}: files not found.")
                continue
            moving_image = ants.image_read(sample.original_sample_path)
            moving_label = ants.image_read(sample.lesion_segmentation_path)

            registration = ants.registration(
                fixed=fixed_image, 
                moving=moving_image, 
                type_of_transform='Rigid'
            )

            registered_mask = ants.apply_transforms(
                fixed=fixed_image,
                moving=moving_label,
                transformlist=registration["fwdtransforms"],
                interpolator="nearestNeighbor"  # Keeps mask strictly binary (0 or 1)
            )

            ants.image_write(registration['warpedmovout'], sample.zoomed_pre_post_path)
            ants.image_write(registered_mask['warpedmovout'], sample.zoomed_segmentation_path)

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

            for n in range(len(mri.lesion_coords)):
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

    def plot_image_registration(self):
        import ants

        out_path = Path(f"/tmp/registration_animation_{self.patient_id}.gif")
        frames = []

        for sample in self.samples:
            img = ants.image_read(sample.zoomed_pre_post_path)
            arr = img.numpy()

            if arr.ndim != 3:
                raise ValueError(f"Expected a 3D image, got shape {arr.shape}")

            z_slice = arr[:, :, arr.shape[2] // 2]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(z_slice, cmap="gray")
            ax.set_title(f"Registered Image for {sample.patient_id}")
            ax.axis("off")
            fig.subplots_adjust(0, 0, 1, 1)

            fig.canvas.draw()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            frame = np.array(PILImage.open(buf).convert("RGB"))
            frames.append(frame)
            plt.close(fig)

        imageio.mimsave(out_path, frames, fps=2, loop=0)
        display(Image(filename=str(out_path)))

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

    def _extract_lesion_data_from_mask(self, labeled_mask, spacing=None, target_shape=None):
        """Extract lesion centroids, sizes, labels, and a common-shape mask from a segmentation mask."""
        labeled_mask = np.asarray(labeled_mask)

        if target_shape is not None:
            target_shape = tuple(int(x) for x in target_shape)
            if labeled_mask.shape != target_shape:
                zoom_factors = [target_shape[i] / labeled_mask.shape[i] for i in range(labeled_mask.ndim)]
                resampled = ndimage.zoom(labeled_mask.astype(float), zoom=zoom_factors, order=0, mode='nearest')
                labeled_mask = np.rint(resampled).astype(np.int32)

        if spacing is None:
            spacing = np.ones(3, dtype=float)
        else:
            spacing = np.asarray(spacing, dtype=float)

        unique_labels = np.unique(labeled_mask)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return (
                np.empty((0, labeled_mask.ndim), dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=int),
                labeled_mask,
            )

        centroids = ndimage.center_of_mass(
            labeled_mask.astype(float),
            labeled_mask,
            unique_labels
        )
        centroids = np.asarray(centroids, dtype=float)
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)

        valid_mask = np.isfinite(centroids).all(axis=1)
        if not np.all(valid_mask):
            valid_idx = np.where(valid_mask)[0]
            unique_labels = unique_labels[valid_idx]
            centroids = centroids[valid_idx]

        centroids = centroids * spacing[:centroids.shape[1]]

        sizes = ndimage.sum(
            np.ones_like(labeled_mask, dtype=np.int32),
            labeled_mask,
            unique_labels
        )
        sizes = np.asarray(sizes, dtype=float)

        return centroids, sizes, unique_labels, labeled_mask

    def _match_lesions_between_scans(self, prev_lesions, curr_lesions, motion_threshold=None):
        """Match lesion instances in consecutive scans using overlap alone."""
        if len(prev_lesions) == 0 or len(curr_lesions) == 0:
            return []

        if motion_threshold is None:
            motion_threshold = float('inf')

        matches = []
        for prev_idx, prev_info in enumerate(prev_lesions):
            prev_mask = prev_info['mask']
            prev_centroid = prev_info['centroid']
            prev_size = prev_info['size']

            best_match = None
            best_score = -1.0
            best_dist = float('inf')

            for curr_idx, curr_info in enumerate(curr_lesions):
                curr_mask = curr_info['mask']
                curr_centroid = curr_info['centroid']
                curr_size = curr_info['size']

                overlap_voxels = int(np.count_nonzero(prev_mask & curr_mask))
                if overlap_voxels <= 0:
                    continue

                overlap_fraction = overlap_voxels / max(prev_size, curr_size)
                centroid_dist = np.linalg.norm(prev_centroid - curr_centroid)

                if centroid_dist > motion_threshold:
                    continue

                score = overlap_fraction - 1e-4 * centroid_dist
                if score > best_score or (np.isclose(score, best_score) and centroid_dist < best_dist):
                    best_score = score
                    best_dist = centroid_dist
                    best_match = curr_idx

            if best_match is not None:
                matches.append((prev_idx, best_match))

        return matches

    def _find_continuation(self, scan_idx, lesion_mask, lesion_centroid, lesion_size,
                            scan_lesions, visited, max_gap=7, motion_threshold=None):
        """Find the next scan that continues the lesion by voxel overlap."""
        if motion_threshold is None:
            motion_threshold = float('inf')

        for future_scan in range(scan_idx + 1, min(scan_idx + 1 + max_gap, len(scan_lesions))):
            if len(scan_lesions[future_scan]['masks']) == 0:
                continue

            best_score = -1.0
            best_dist = float('inf')
            best_idx = None

            for lesion_idx in range(len(scan_lesions[future_scan]['masks'])):
                if (future_scan, lesion_idx) in visited:
                    continue

                future_mask = scan_lesions[future_scan]['masks'][lesion_idx]
                future_centroid = scan_lesions[future_scan]['centroids'][lesion_idx]
                future_size = scan_lesions[future_scan]['sizes'][lesion_idx]

                overlap_voxels = int(np.count_nonzero(lesion_mask & future_mask))
                if overlap_voxels <= 0:
                    continue

                overlap_fraction = overlap_voxels / max(lesion_size, future_size)
                centroid_dist = np.linalg.norm(lesion_centroid - future_centroid)
                if centroid_dist > motion_threshold:
                    continue

                score = overlap_fraction - 1e-4 * centroid_dist
                if score > best_score or (np.isclose(score, best_score) and centroid_dist < best_dist):
                    best_score = score
                    best_dist = centroid_dist
                    best_idx = lesion_idx

            if best_idx is not None:
                return future_scan, best_idx

        return None, None

    def _build_overlap_trajectories(self, scan_lesions, num_scans, max_gap=7, motion_threshold=None):
        """Build lesion trajectories by following overlap-only matches with optional gaps."""
        visited = set()
        trajectories = []

        for scan_idx in range(num_scans):
            for lesion_idx in range(len(scan_lesions[scan_idx]['labels'])):
                if (scan_idx, lesion_idx) in visited:
                    continue

                trajectory = {
                    'scan_indices': [scan_idx],
                    'labels': [(scan_idx, scan_lesions[scan_idx]['labels'][lesion_idx])],
                    'sizes': [scan_lesions[scan_idx]['sizes'][lesion_idx]],
                    'centroids': [scan_lesions[scan_idx]['centroids'][lesion_idx]],
                    'masks': [scan_lesions[scan_idx]['masks'][lesion_idx]]
                }
                visited.add((scan_idx, lesion_idx))

                current_scan = scan_idx
                current_lesion_idx = lesion_idx
                current_mask = scan_lesions[scan_idx]['masks'][lesion_idx]
                current_centroid = scan_lesions[scan_idx]['centroids'][lesion_idx]
                current_size = scan_lesions[scan_idx]['sizes'][lesion_idx]

                while current_scan < num_scans - 1:
                    next_scan, next_lesion_idx = self._find_continuation(
                        current_scan, current_mask, current_centroid, current_size,
                        scan_lesions, visited, max_gap=max_gap, motion_threshold=motion_threshold
                    )

                    if next_scan is None:
                        break

                    trajectory['scan_indices'].append(next_scan)
                    trajectory['labels'].append((next_scan, scan_lesions[next_scan]['labels'][next_lesion_idx]))
                    trajectory['sizes'].append(scan_lesions[next_scan]['sizes'][next_lesion_idx])
                    trajectory['centroids'].append(scan_lesions[next_scan]['centroids'][next_lesion_idx])
                    trajectory['masks'].append(scan_lesions[next_scan]['masks'][next_lesion_idx])

                    visited.add((next_scan, next_lesion_idx))
                    current_scan = next_scan
                    current_lesion_idx = next_lesion_idx
                    current_mask = scan_lesions[next_scan]['masks'][next_lesion_idx]
                    current_centroid = scan_lesions[next_scan]['centroids'][next_lesion_idx]
                    current_size = scan_lesions[next_scan]['sizes'][next_lesion_idx]

                trajectories.append(trajectory)

        return trajectories

    def merge_lesion_to_trajectory(self, max_gap=7, spatial_threshold=None,
                                    size_ratio_threshold=None, distance_threshold=None):
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


        scan_volumes = []
        scan_affines = []
        scan_shapes = []
        for sample in self.samples:
            volume_data, affine = sample.load_mri(zoomed=True, affine=True)
            scan_volumes.append(volume_data)
            scan_affines.append(affine)
            scan_shapes.append(volume_data.shape)

        volume = np.count_nonzero(scan_volumes[0] > 0) if scan_volumes else 0

        reference_shape = tuple(max(shape[d] for shape in scan_shapes) for d in range(len(scan_shapes[0]))) if scan_shapes else None
        if reference_shape is None:
            print("No scan data available; skipping trajectory building.")
            return

        if spatial_threshold is None:
            motion_threshold = 0.07 * max(reference_shape)
        else:
            motion_threshold = float(spatial_threshold)

        scan_lesions = []
        lesion_counts = []
        for i, sample in enumerate(self.samples):
            labeled_mask, affine = sample.load_mri_segmentation(zoomed=True, affine=True)
            spacing = np.abs(np.diag(affine)[:3]).astype(float)
            centroids, sizes, labels, aligned_mask = self._extract_lesion_data_from_mask(
                labeled_mask,
                spacing=spacing,
                target_shape=reference_shape
            )
            num_lesions = len(labels)
            lesion_counts.append(num_lesions)

            lesion_masks = [(aligned_mask == label).astype(np.uint8) for label in labels]
            scan_lesions.append({
                'centroids': centroids,
                'sizes': sizes,
                'labels': labels,
                'labeled_mask': aligned_mask.astype(np.uint8),
                'masks': lesion_masks,
                'spacing': spacing
            })
            print(f"  Scan {i} ({self.samples[i].date}): {num_lesions} lesions")


        trajectories = self._build_overlap_trajectories(
            scan_lesions,
            len(self.samples),
            max_gap=max_gap,
            motion_threshold=motion_threshold
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

        labeled_mask_cache = [np.zeros(shape, dtype=np.uint8) for shape in scan_shapes]
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
                overlap_mask = (lesion_mask > 0) & (labeled_mask_cache[scan_idx] != 0)
                if np.any(overlap_mask):
                    print(f"  Warning: trajectory {traj_id + 1} overlaps existing voxels in scan {scan_idx}; skipping overlapping voxels")
                    lesion_mask[overlap_mask] = 0

                if np.any(lesion_mask > 0):
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
            img = nib.Nifti1Image(labeled_mask_cache[i], scan_affines[i])
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