"""Supplement the immutable coefficient probe with amplitude-weighting control."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PIN="9b3d3a43e94eeb259cbd8f4c705e69276c6e268830d274338adaa32804d67ec3"
if hashlib.sha256((HERE/'bicoherence_math_probe_v1.py').read_bytes()).hexdigest()!=PIN:
    raise ValueError('original probe changed')
from bicoherence_math_probe_v1 import estimate


def main():
    root=HERE.parent/'results/bicoherence_amplitude_probe_v1'
    if root.exists(): raise ValueError('exclusive output required')
    x=np.zeros((128,129),dtype=np.complex128)
    x[:,[4,7,11]]=1
    x[64:,11]=10
    measured=estimate(x)
    expected=11**2/(2*101)
    np.testing.assert_allclose(measured['squared_bicoherence'],expected,rtol=1e-14,atol=1e-14)
    closure=np.angle(x[:,4]*x[:,7]*x[:,11].conj())
    np.testing.assert_array_equal(closure,np.zeros(128))
    # Energy-weighted effective count for this particular u and v design.
    # This is descriptive, not a universal independent-sample correction.
    result={'status':'synthetic_amplitude_counterexample_complete','realizations':128,
        'constant_biphase_radians':0.,'expected_squared_bicoherence':expected,
        'measured':measured,'independent_phase_only_resultant_length':float(abs(np.mean(np.exp(1j*closure)))),
        'formula':'(64*1+64*10)^2 / (128*(64*1^2+64*10^2)) = 121/202',
        'interpretation':'Pooled bicoherence depends on cross-frame amplitudes as well as phase closure.',
        'external_validation_passed':False,'classifier_admitted':False,'classifier_fits':0,
        'source_probe_sha256':PIN,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    root.mkdir()
    np.savez(root/'synthetic_coefficients.npz',spectra=x)
    (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    products={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir()}
    (root/'COMMIT.json').write_text(json.dumps({'status':'committed','products':products},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
