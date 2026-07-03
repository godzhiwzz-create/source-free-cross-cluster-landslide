from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr


def test_official_netcdf_preprocessing_contract(tmp_path):
    s2_root = tmp_path / "s2"
    output = tmp_path / "npz"
    s2_root.mkdir()
    time = np.arange(4).astype("datetime64[D]")
    shape = (4, 4, 4)
    variables = {
        band: (("time", "y", "x"), np.full(shape, 1000, dtype=np.int16))
        for band in (
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        )
    }
    variables.update(
        {
            "SCL": (("time", "y", "x"), np.full(shape, 4, dtype=np.int16)),
            "MASK": (("time", "y", "x"), np.zeros(shape, dtype=np.uint8)),
            "DEM": (
                ("time", "y", "x"),
                np.full(shape, 500, dtype=np.float32),
            ),
        }
    )
    dataset = xr.Dataset(variables, coords={"time": time})
    dataset["MASK"].values[:, 1:3, 1:3] = 1
    source = s2_root / "chimanimani_s2_0001.nc"
    dataset.to_netcdf(source)

    root = Path(__file__).parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "preprocess_sen12.py"),
            "--s2-root",
            str(s2_root),
            "--output-dir",
            str(output),
            "--cluster-map",
            str(root / "configs" / "region_to_cluster.json"),
        ],
        check=True,
    )

    files = list(output.glob("Africa__*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as archive:
        assert archive["X"].shape == (1, 4, 14, 4, 4)
        assert archive["Y"].shape == (1, 4, 4)
        assert np.allclose(archive["X"][0, :, 0], 0.1)
        assert np.allclose(archive["X"][0, :, 10:12], 0.0)
        assert np.allclose(archive["X"][0, :, 12], 0.5)
