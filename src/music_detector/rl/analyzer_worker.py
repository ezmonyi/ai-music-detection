"""One frozen analysis call in its own Python/CUDA runtime; no auto setup.

JSON on stdin/stdout is the only protocol. Library progress goes to stderr.
The process exits after each call so analysis cannot retain GPU allocations
while the policy replay backpropagates. Seeded shifts are a newly declared RL
measurement protocol, not bit-identical replay of historical unseeded stems.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from .rewards import ArtifactReward, FEATURE_IMPLEMENTATION_SHA256


def execute(request: dict) -> dict:
    families = request["families"]
    # This verifies frozen algorithms and detector assets in THIS runtime,
    # not just in the training process. Neural readiness is checked below.
    frozen = ArtifactReward(families, request["bundle_sha256"],
                            analyzer=lambda *a, **kw: {})
    neural = [f for f in frozen.families if f in {"S", "D", "R", "P"}]
    from ..neural import runtime_status, _device
    status = (runtime_status(families=neural, cache_dir=request["cache_dir"])
              if neural else {"ready": True, "families": [], "versions": {}})
    if neural and not status["ready"]:
        raise RuntimeError("Frozen analyzer is not ready: " + "; ".join(status["errors"]))
    device = _device(request["device"])
    receipt = {
        "backend": "isolated_frozen_analyzer_v1",
        "families": list(frozen.families), "device": device,
        "seed": request["seed"], "bundle_sha256": frozen.bundle_sha256,
        "feature_implementation_sha256": dict(FEATURE_IMPLEMENTATION_SHA256),
        "runtime": status,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if request["action"] == "status":
        return {"receipt": receipt}
    if request["action"] != "analyze":
        raise ValueError("Unknown analysis worker action")
    import torch
    random.seed(request["seed"])
    np.random.seed(request["seed"] % 2**32)
    torch.manual_seed(request["seed"])
    torch.set_num_threads(4)
    from ..analysis import analyze
    result = analyze(Path(request["path"]), list(frozen.families), device=device)
    result["isolated_analyzer_receipt"] = receipt
    return {"receipt": receipt, "result": result}


def main() -> None:
    request = json.loads(sys.stdin.read())
    with redirect_stdout(sys.stderr):
        response = execute(request)
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
