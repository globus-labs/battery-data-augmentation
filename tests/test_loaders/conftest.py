from pathlib import Path

import pandas as pd
from pytest import fixture

_file_dir = Path(__file__).parent / '..' / 'files'


@fixture()
def example_data() -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Load in the example data"""

    cycle_data = [pd.read_csv(f) for f in sorted(_file_dir.joinpath('example-data').glob('discharge*csv.gz'))]
    summary_data = [pd.read_csv(f) for f in sorted(_file_dir.joinpath('example-data').glob('summary*csv.gz'))]
    return cycle_data, summary_data
