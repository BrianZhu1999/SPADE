# Standing-wave tube measurements

Calibrated experimental pressure and axial particle velocity used in the SPADE study. The 385 Hz record supplies equation discovery and tests on later observations. The separately measured 395 Hz record supplies the frequency-transfer test.

## Files and units

| File | Contents |
| --- | --- |
| `record_385Hz.npz`, `record_395Hz.npz` | Complete 5 s calibrated records at the 12 valid measured probe positions |
| `observations.npz`, `metadata.json` | Unchanged 385 Hz search input and its original partition metadata |
| `records.json` | Units, coordinates, record hashes and evaluation intervals |
| `calibration/calibration_385Hz.json`, `calibration/calibration_395Hz.json` | Original calibration manifests, including complex correction factors and source hashes |
| `calibration/Tube*.xls` | The 12 probe-response workbooks used for calibration |
| `SHA256SUMS` | SHA-256 checksums of all other files in this directory |

Each complete-record archive contains `field_p` (pressure in Pa), `field_v` (axial particle velocity in m/s), `t` (time in s), `x` (relative axial position in m), and `sensor_ids_zero_based`. Both fields have shape `(1, 50000, 12)` and retain the original float64 values. The sampling frequency is 10,000 Hz. The retained probe indices are `[1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]`; relative positions equal these indices multiplied by 0.08 m, increasing toward the rigid end. The gaps between measured probes are preserved.

```python
from pathlib import Path
import numpy as np

root = Path("data/StandingWaveTube")  # From the SPADE repository root.
with np.load(root / "record_385Hz.npz", allow_pickle=False) as data:
    pressure = data["field_p"][0]  # (50000 time samples, 12 probes), Pa
    velocity = data["field_v"][0]  # Same shape, m/s
    time_s = data["t"]
    position_m = data["x"]
```

## Data partitions

All sample and block indices below are zero-based, with an excluded upper endpoint: `[100, 6100)` contains 6,000 samples.

The 385 Hz search uses samples `[100, 6100)`, reshaped into 30 consecutive blocks of 200 samples. Blocks `[0, 18)` are used for training, `[18, 24)` for validation, and `[24, 30)` for testing. `observations.npz` contains `field_p` and `field_v` with shape `(30, 200, 12)`, a 200-sample within-block time coordinate, and the 12 measured spatial coordinates. These blocks come from one experimental recording.

The later 385 Hz weak-equation test uses samples `[10000, 11200)`. The independent 395 Hz weak-equation test uses `[100, 1300)`. Forward comparisons use `[10000, 10202)` at 385 Hz and `[100, 302)` at 395 Hz: one initial frame and 201 future frames. Measured incoming boundary characteristics are supplied at the two endpoint probes; prediction errors are evaluated at the ten interior probes. Equations and coefficients are held fixed from the 385 Hz search.

## Calibration and provenance

Calibration uses the 2025-12-24 response tables at the U calibration position. Sheet1 stores pressure sensitivity in mV/Pa; Sheet2 stores velocity sensitivity in V/(m/s). Both store phase in radians. Positive-frequency pressure spectra are multiplied by `1000/S_p * exp(-1j*phi_p)`. Velocity spectra are multiplied by `1/S_v * exp(1j*(pi/2-phi_v))`, accounting for the standing-wave phase included in the original velocity-response tables. Here `S_p` and `S_v` are the tabulated sensitivities and `phi_p` and `phi_v` the tabulated phases. The manifests record each probe's correction factor at the corresponding excitation frequency. Calibration precedes resampling from 12,800 to 10,000 Hz.

The complete-record archives select the 12 valid probes from the original calibrated arrays and compress them without changing their values. No interpolation, denoising or additional amplitude scaling is applied. The original arrays, raw-voltage records and calibration workbooks are identified by SHA-256 in the manifests. Historical server paths in those manifests record provenance; reading the published archives requires only NumPy.
