import numpy as np

from .dataloader import LesionDataset


class FourTimepointLesionDataset(LesionDataset):
    """Sample four ordered scan times from each lesion for Neural ODE training.

    Each item returns coordinates with shape [4, N, 4], labels with shape
    [4, N], and one lesion embedding id. Four distinct training dates are
    selected at random and kept in chronological order.
    """

    num_timepoints = 3

    def _training_dates(self, trj):
        dates = list(trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates)
        lesion_id = f"{trj.patient_id}_{trj.label_id}"

        if lesion_id in self.validation_samples:
            validation = self.validation_samples[lesion_id]
            extrapolation_dates = validation["extrapolation_dates"]
            dates = [date for date in dates if date not in extrapolation_dates]
            dates.pop(int(validation["interpolation"]))

        return sorted(dates, key=str)

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        dates = self._training_dates(trj)
        if len(dates) < self.num_timepoints:
            raise ValueError(
                f"Lesion {trj.patient_id}_{trj.label_id} has {len(dates)} "
                f"training scans; {self.num_timepoints} are required."
            )

        selected_indices = np.sort(
            np.random.choice(len(dates), size=self.num_timepoints, replace=False)
        )
        selected_dates = [str(dates[index]) for index in selected_indices]

        samples = [self.prepare_data(date, trj) for date in selected_dates]
        coords = np.stack([sample[0] for sample in samples], axis=0)
        labels = np.stack([sample[1] for sample in samples], axis=0)

        return coords, labels, self.get_embedding_id(trj)
