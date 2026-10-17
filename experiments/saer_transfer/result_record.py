"""Write one machine-readable experiment result."""

import json
from pathlib import Path


def write_metrics(path, mse, mae, model, best_epoch):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mse": float(mse),
        "mae": float(mae),
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch": None if best_epoch is None else int(best_epoch),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
