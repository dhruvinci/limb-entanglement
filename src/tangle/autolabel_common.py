"""Shared paths and mask encoding for the Astra + SAM3.1 auto-labeller (stdlib + numpy only)."""
import hashlib
import json
from pathlib import Path

from paths import WORKSPACE as ROOT  # noqa: E402
EVIDENCE = ROOT/'evidence/trial100-v1'
OUT = ROOT/'autolabel'
from paths import SAM31_CKPT  # noqa: E402,F401
TAXONOMY = ROOT/'docs/autolabel/taxonomy.v1.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def evidence_root(trial_id):
    """trial100 sources (trialNNN), the autolabel-v2 breadth cohort (a2sNNN), continuous matches (a3mNNN)."""
    if trial_id.startswith('a2s'):
        return ROOT/'evidence/autolabel-v2'
    if trial_id.startswith('a3m'):
        return ROOT/'evidence/autolabel-v3'
    if trial_id.startswith('a4g'):
        return ROOT/'evidence/autolabel-v4'
    return EVIDENCE


def packet(trial_id, packet_id):
    path = evidence_root(trial_id)/trial_id/'selected-v1'/packet_id/'packet.json'
    return json.loads(path.read_text()), path


def interval_dir(run, trial_id, packet_id):
    return OUT/run/trial_id/packet_id


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def rle_encode(mask):
    """COCO uncompressed RLE (column-major counts, starting with zeros)."""
    import numpy as np
    flat = np.asarray(mask, dtype=bool).flatten(order='F')
    changes = np.flatnonzero(np.diff(np.concatenate([[False], flat, [False]]).astype(np.int8)))
    runs = np.diff(np.concatenate([[0], changes, [flat.size]]))
    counts = runs.tolist()
    if counts and counts[-1] == 0:
        counts.pop()
    return {'size': [int(mask.shape[0]), int(mask.shape[1])], 'counts': counts}


def rle_decode(rle):
    import numpy as np
    h, w = rle['size']
    flat = np.zeros(h*w, dtype=bool)
    pos, value = 0, False
    for n in rle['counts']:
        if value:
            flat[pos:pos+n] = True
        pos += n
        value = not value
    return flat.reshape((h, w), order='F')


def mask_box(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix*iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0
