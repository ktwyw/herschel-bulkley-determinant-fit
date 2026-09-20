import json
from pathlib import Path

import numpy as np

from hb_mullineux.cli import main, read_csv
from hb_mullineux import synthetic_data


def test_cli_reads_header_csv_and_writes_json(tmp_path: Path) -> None:
    x, y = synthetic_data(seed=1)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("shear_rate_1_s,shear_stress_Pa\n" + "\n".join(f"{a},{b}" for a, b in zip(x, y)))
    out = tmp_path / "fit.json"
    assert main([str(csv_path), "--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert 0.3 < payload["n"] < 0.45


def test_read_csv_semicolon_decimal_comma_and_no_header(tmp_path: Path) -> None:
    p = tmp_path / "eu.csv"
    p.write_text("1;7,5\n2;9,1\n4;11,3\n8;14,0\n")
    x, y, labels = read_csv(p)
    assert np.allclose(x, [1, 2, 4, 8]) and np.allclose(y, [7.5, 9.1, 11.3, 14.0])
    assert labels == ("column 1", "column 2")
