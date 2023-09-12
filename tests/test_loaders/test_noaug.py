from bataug.loaders.noaug import FixedCycleDataset


def test_fixed(example_data):
    cycle_data, summary_data = example_data
    loader = FixedCycleDataset(cycles=cycle_data, summaries=summary_data, batch_size=2, train_cycles=2)
    for inputs, outputs, outputs_mask in loader:
        assert inputs.shape == (2, 2, 128)
        assert outputs.shape[0] == 2
        assert outputs_mask.all()
