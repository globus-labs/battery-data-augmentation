"""Baseline data loader for a class which does not use augmentation"""

from more_itertools import grouper
import numpy as np

from .base import BaseBatteryDataset


class FixedCycleDataset(BaseBatteryDataset):
    """Dataset which always uses the same number of input cycles

    Args:
        train_cycles: Number of cycles to for training
        cycles: Discharge curves for each battery in a dataset in BEEP format
        summaries: Summary of each cycle for each battery in BEEP format
        batch_size: Number of batteries to use per batch
        voltage_range: Minimum and maximum voltage for the input features
        voltage_steps: Number of voltage steps for the input vary
        random_seed: Random seed for generating batches
    """

    def __init__(self, train_cycles: int, **kwargs):
        super().__init__(**kwargs)
        self.train_cycles = train_cycles

    def __iter__(self):
        # Short the cells in random order
        inds = np.arange(len(self.cycles))
        self.rng.shuffle(inds)

        # Generate the batches of training data
        for batch_inds in grouper(inds, self.batch_size, incomplete='ignore'):  # Do not generate incomplete batches
            # Assemble the input images
            inputs = np.zeros((self.batch_size, self.train_cycles, self.voltage_steps))
            for i, ind in enumerate(batch_inds):
                inputs[i, :, :] = self.generate_input_image(ind, self.train_cycles)

            # Assemble the outputs
            discharge_curves = [
                self.generate_output_vector(ind, self.train_cycles) for ind in batch_inds
            ]
            outputs = np.zeros((self.batch_size, max(map(len, discharge_curves))))
            output_mask = np.zeros_like(outputs, dtype=bool)
            for i, curve in enumerate(discharge_curves):
                output_mask[i, :len(curve)] = True
                outputs[i, :len(curve)] = curve

            yield inputs, outputs, output_mask
