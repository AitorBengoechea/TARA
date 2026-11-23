import os
import pytest
import pandas as pd
from input_output.exporter import export_ensemble_results

class DummyEnsembleData:
    """
    Duck-typed pure state container to test IO without importing from the analyzer domain.
    """
    def __init__(self):
        """Initialize the instance and its required state.
        """
        self.xs_inelastic = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], 
            index=[0, 1], columns=[0.01, 0.1]
        )
        self.scattering_kernel = {
            0.0253: pd.DataFrame(
                [[0.5, 0.5], [0.6, 0.4]], 
                index=[0, 1], columns=[0.01, 0.1]
            )
        }
        self.xs_total = None  # Simulates an uncomputed property

def test_export_ensemble_results_hdf5(tmp_path):
    """
    Test hierarchical HDF5 serialization of ensemble results.

    Verifies that export_ensemble_results:
    - Serializes standard 2D DataFrames to the root HDF5 path (e.g., /xs_inelastic).
    - Serializes Multi-Ein dictionaries into nested HDF5 paths (e.g., /scattering_kernel/ein_0_0253).
    - Ignores unpopulated (None) properties during serialization.
    """
    data = DummyEnsembleData()
    output_file = tmp_path / "test_results.h5"
    
    export_ensemble_results(data, str(output_file))
    
    assert os.path.exists(output_file)
    
    # Verify the HDF5 internal hierarchy
    with pd.HDFStore(output_file, mode='r') as store:
        keys = store.keys()
        
        # Verify 2D DataFrame path
        assert '/xs_inelastic' in keys
        retrieved_inelastic = store['xs_inelastic']
        pd.testing.assert_frame_equal(retrieved_inelastic, data.xs_inelastic)
        
        # Verify Multi-Ein hierarchical path (dots replaced with underscores)
        assert '/scattering_kernel/ein_0_0253' in keys
        retrieved_kernel = store['scattering_kernel/ein_0_0253']
        pd.testing.assert_frame_equal(retrieved_kernel, data.scattering_kernel[0.0253])
        
        # Verify empty properties are skipped
        assert '/xs_total' not in keys
