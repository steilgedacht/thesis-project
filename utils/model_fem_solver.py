import gc
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import zoom
from dolfinx import mesh, fem, default_scalar_type
import dolfinx.fem.petsc  # noqa: F401 - required for fem.petsc.LinearProblem to exist
from mpi4py import MPI
import ufl


# ---------------------------------------------------------------------------
# Crop / pad / resample helpers
# ---------------------------------------------------------------------------

def _compute_padded_bbox(mask, padding_frac=0.5, min_pad_voxels=(4, 4, 4)):
    """
    Bounding box of the nonzero region of `mask`, expanded by `padding_frac`
    of its own extent in each axis (with a minimum pad in voxels), and
    clipped to the array bounds.
    """
    nonzero = np.argwhere(mask > 0)
    if nonzero.size == 0:
        raise ValueError("Mask is empty - cannot compute a lesion bounding box.")

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1  # exclusive upper bound

    extents = maxs - mins
    pad = np.maximum((extents * padding_frac).astype(int), np.array(min_pad_voxels))

    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(mask.shape))

    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def _resample_to_shape(volume, target_shape):
    """Resample a 3D volume onto `target_shape` with linear interpolation."""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resampled = zoom(volume, zoom_factors, order=1, mode="nearest")
    # scipy's zoom output shape can be off by a voxel due to rounding
    out = np.zeros(target_shape, dtype=volume.dtype)
    slices = tuple(slice(0, min(t, r)) for t, r in zip(target_shape, resampled.shape))
    out[slices] = resampled[slices]
    return out


def crop_and_resample_trajectory(trajectory, target_shape=(32, 32, 32),
                                  padding_frac=0.5, voxel_spacing=None):
    """
    Takes a `Lesion_Trajectory` instance, loads every scan's mask via
    `trajectory.load_labels_for_inr(absolute_day_number=True, affine=True)`
    (called exactly ONCE - that method hits disk per date, so we don't want
    to call it repeatedly), crops every scan to a padded bounding box
    around the LESION AT t0 ONLY (so no information from later/future
    scans leaks into the crop), and resamples each crop onto a fixed
    `target_shape` grid.

    `absolute_day_number=True` is important: without it, load_labels_for_inr
    returns time normalized to [-1, 1] over the trajectory's own span,
    which is useless for computing a physical `dt` in days for the FEM
    solver. With it, times are real day-offsets from the first scan.

    This also makes the FEM mesh resolution and the data resolution match
    exactly, which removes the need to interpolate the coarse FEM solution
    back up onto full native MRI resolution on every simulate() call - that
    up-interpolation was previously the dominant memory/time cost.

    Parameters
    ----------
    trajectory : a Lesion_Trajectory instance
    target_shape : fixed grid shape everything gets resampled to
    padding_frac : extra margin around the t0 lesion bbox, as a fraction
        of the lesion's own extent in each axis. Needs to be generous
        enough to contain growth out to your longest extrapolation horizon
        - a lesion that grows past the crop edge will be clipped.
    voxel_spacing : (sx, sy, sz) physical size of one voxel in mm. If None
        (default), it's derived automatically from the affine matrix
        returned by load_labels_for_inr(affine=True) - pass an explicit
        value only to override that.

    Returns
    -------
    dict with:
      "times"         : list of real day-offsets from the first scan
      "masks"         : list of resampled (target_shape) masks, same order
      "domain_bounds" : physical (x, y, z) size in mm of the crop box -
                        pass this to FEMLesionGrowthSolver(domain_bounds=...)
      "crop_slices"   : slices used to crop the original volume (kept for
                        debugging / mapping predictions back if needed)
      "voxel_spacing" : the (derived or overridden) voxel spacing used
    """
    samples, affine_mat = trajectory.load_labels_for_inr(
        absolute_day_number=True, affine=True
    )

    if len(samples) < 3:
        raise ValueError(
            f"Trajectory (patient={getattr(trajectory, 'patient_id', '?')}, "
            f"label={getattr(trajectory, 'label_id', '?')}) has fewer than "
            f"3 loadable scans."
        )

    if voxel_spacing is None:
        # column norms of the affine's rotation/scale block = mm per voxel
        voxel_spacing = tuple(float(np.linalg.norm(affine_mat[:3, i])) for i in range(3))

    mask_t0 = samples[0][0]
    crop_slices = _compute_padded_bbox(mask_t0, padding_frac=padding_frac)
    crop_shape_voxels = tuple(s.stop - s.start for s in crop_slices)
    domain_bounds = tuple(crop_shape_voxels[i] * voxel_spacing[i] for i in range(3))

    times, masks = [], []
    for mask, day_offset in samples:
        cropped = mask[crop_slices]
        resampled = _resample_to_shape(cropped, target_shape)
        times.append(day_offset)
        masks.append(resampled)

    return {
        "times": times,
        "masks": masks,
        "domain_bounds": domain_bounds,
        "crop_slices": crop_slices,
        "voxel_spacing": voxel_spacing,
    }


# ---------------------------------------------------------------------------
# FEM solver
# ---------------------------------------------------------------------------

class FEMLesionGrowthSolver:
    """
    Memory-safe version: mesh, function space, the Delaunay triangulation of
    the dof coordinates, target-grid coordinates, and the variational
    problem are all built ONCE in __init__ and reused across every
    simulate() call - instead of being rebuilt from scratch on every one of
    the ~30+ calls per trajectory that Nelder-Mead fitting triggers.

    If you're feeding it crops produced by crop_and_resample_trajectory()
    with target_shape == target_grid_shape, the mesh and the input/output
    data are already the same resolution, so the up/down interpolation in
    _map_numpy_to_fenics / _map_fenics_to_numpy stays cheap.
    """

    def __init__(self, target_grid_shape=(32, 32, 32), domain_bounds=(1.0, 1.0, 1.0)):
        self.target_grid_shape = target_grid_shape
        self.domain_bounds = domain_bounds

        self.domain = mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0, 0.0]), np.array(domain_bounds)],
            list(target_grid_shape),
        )
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

        _t0 = time.perf_counter()
        self._dof_coords = self.V.tabulate_dof_coordinates()[:, :3]
        # qhull_options='QJ' (joggle) perturbs the points slightly before
        # triangulating. Without it, Qhull can be pathologically slow on
        # perfectly regular/axis-aligned lattices like box-mesh vertices,
        # since many points are exactly cospherical/coplanar and Qhull has
        # to work much harder to break ties.
        self._tri = Delaunay(self._dof_coords, qhull_options='QJ')
        print(f"[FEMLesionGrowthSolver] Delaunay triangulation took {time.perf_counter() - _t0:.3f}s "
              f"for {len(self._dof_coords)} points")

        # cache target-grid coordinates per voxel shape instead of
        # rebuilding the meshgrid on every call
        self._target_coord_cache = {}

        # --- build the variational problem ONCE ---
        self.c_n = fem.Function(self.V)
        self.D = fem.Constant(self.domain, default_scalar_type(0.0))
        self.rho = fem.Constant(self.domain, default_scalar_type(0.0))
        self.dt_const = fem.Constant(self.domain, default_scalar_type(1.0))

        c = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        F = (
            (c - self.c_n) / self.dt_const * v * ufl.dx
            + self.D * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx
            - self.rho * self.c_n * (1.0 - self.c_n) * v * ufl.dx
        )
        a, L = ufl.system(F)
        # built once; problem.solve() reassembles A/b against the current
        # D / rho / dt_const values internally each time it's called.
        # petsc_options_prefix namespaces this problem's PETSc options so
        # multiple LinearProblem instances (one per trajectory) don't clash;
        # id(self) keeps it unique per solver instance.
        self.problem = fem.petsc.LinearProblem(
            a, L, bcs=[],
            petsc_options_prefix=f"fem_lesion_solver_{id(self)}_",
        )

    def _map_numpy_to_fenics(self, mask_3d):
        nx, ny, nz = mask_3d.shape
        bx, by, bz = self.domain_bounds
        x = np.linspace(0, bx, nx)
        y = np.linspace(0, by, ny)
        z = np.linspace(0, bz, nz)
        interpolator = RegularGridInterpolator(
            (x, y, z), mask_3d, bounds_error=False, fill_value=0.0
        )
        self.c_n.x.array[:] = interpolator(self._dof_coords)

    def _get_target_coords(self, shape):
        if shape not in self._target_coord_cache:
            nx, ny, nz = shape
            bx, by, bz = self.domain_bounds
            x = np.linspace(0, bx, nx)
            y = np.linspace(0, by, ny)
            z = np.linspace(0, bz, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            self._target_coord_cache[shape] = coords
        return self._target_coord_cache[shape]

    def _map_fenics_to_numpy(self, c_h, original_shape):
        target_coords = self._get_target_coords(original_shape)
        # reuse the cached triangulation instead of recomputing Delaunay
        interp = LinearNDInterpolator(self._tri, c_h.x.array, fill_value=0.0)
        return interp(target_coords).reshape(original_shape)

    def simulate(self, c_init_array, D_val, rho_val, dt, num_steps, _profile=False):
        t0 = time.perf_counter() if _profile else None
        self._map_numpy_to_fenics(c_init_array)
        t1 = time.perf_counter() if _profile else None

        self.D.value = D_val
        self.rho.value = rho_val
        self.dt_const.value = dt

        c_h = None
        for _ in range(num_steps):
            c_h = self.problem.solve()
            self.c_n.x.array[:] = c_h.x.array[:]
        t2 = time.perf_counter() if _profile else None

        result = self._map_fenics_to_numpy(c_h, c_init_array.shape)
        t3 = time.perf_counter() if _profile else None

        if _profile:
            print(
                f"[simulate] to_fenics={t1 - t0:.3f}s  "
                f"solve_loop({num_steps} steps)={t2 - t1:.3f}s  "
                f"to_numpy={t3 - t2:.3f}s  total={t3 - t0:.3f}s"
            )

        return result

    def close(self):
        """Explicitly release PETSc C-level objects instead of relying on
        the GC. Call this once you're done with a solver instance, e.g.
        after finishing all simulate() calls for one trajectory."""
        for attr in ("A", "b"):
            obj = getattr(self.problem, attr, None)
            if obj is not None:
                obj.destroy()
        solver = getattr(self.problem, "solver", None)
        if solver is not None:
            solver.destroy()
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass