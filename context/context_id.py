import hashlib
import json

def make_context_id(dataset_id, mode, columns):
    payload = {
        "dataset_id": dataset_id,
        "mode": mode,
        "columns": sorted(columns) if columns else None
    }

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()
