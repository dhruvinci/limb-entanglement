"""Harmony4D (MIT, close human interaction) as a ground-truth benchmark for tangle ownership.

Harmony4D ships per-frame SMPL fits (vertices in world coordinates) for both people and calibrated static exo cameras
(COLMAP, OPENCV_FISHEYE), but no masks. Ground-truth ownership maps are rendered here: both bodies' triangles,
projected with the fisheye model, painted far-to-near (painter's algorithm), so the nearer body owns each pixel.
SMPL is an unclothed body: boundaries are tighter than the real silhouette (hair, shorts, gloves), but who owns a
region is exact up to fit error, which is what the tangle benchmark measures.
World -> COLMAP is inv(scale.npy) (verified: projected vertex boxes equal processed_data/bbox to the pixel).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from paths import H4D
FACES = np.load(H4D/'smpl_faces.npy')
PEOPLE = ['aria01', 'aria02']


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class Sequence:
    def __init__(self, root):
        self.root = Path(root)
        ws = self.root/'colmap/workplace'
        self.cams = {}
        for line in open(ws/'cameras.txt'):
            if not line.startswith('#'):
                p = line.split()
                self.cams[int(p[0])] = (int(p[2]), int(p[3]), np.array(list(map(float, p[4:]))))
        self.poses = {}
        lines = [l for l in open(ws/'images.txt') if not l.startswith('#')]
        for line in lines[::2]:
            p = line.split()
            if len(p) >= 10 and p[9].startswith('cam'):
                self.poses.setdefault(p[9].split('/')[0], (qrot(np.array(list(map(float, p[1:5])))),
                                                             np.array(list(map(float, p[5:8]))), int(p[8])))
        self.world_to_colmap = np.linalg.inv(np.load(ws/'scale.npy'))
        self.frames = sorted(int(f.stem) for f in (self.root/'processed_data/smpl').glob('*.npy'))

    def image_path(self, cam, frame):
        return self.root/'exo'/cam/'images'/f'{frame:05d}.jpg'

    def project(self, X, cam):
        R, t, cid = self.poses[cam]
        _, _, P = self.cams[cid]
        fx, fy, cx, cy, k1, k2, k3, k4 = P
        Xh = np.c_[X, np.ones(len(X))] @ self.world_to_colmap.T
        Xc = Xh[:, :3] @ R.T + t
        z = Xc[:, 2]
        x, y = Xc[:, 0]/z, Xc[:, 1]/z
        r = np.sqrt(x*x + y*y)
        th = np.arctan(r)
        thd = th*(1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
        s = np.where(r > 1e-8, thd/np.maximum(r, 1e-8), 1.0)
        return np.stack([fx*s*x + cx, fy*s*y + cy], 1), z

    def gt(self, cam, frame, size=(1280, 720)):
        """Label map at `size` (w, h): 0 background, 1 aria01, 2 aria02 (nearer body wins)."""
        d = np.load(self.root/'processed_data/smpl'/f'{frame:05d}.npy', allow_pickle=True).item()
        R, t, cid = self.poses[cam]
        W, H = self.cams[cid][:2]
        sx, sy = size[0]/W, size[1]/H
        tris = []
        for n, who in enumerate(PEOPLE, start=1):
            uv, z = self.project(d[who]['vertices'], cam)
            uv = uv*[sx, sy]
            f = FACES
            depth = z[f].mean(1)
            ok = (z[f] > 0).all(1)
            for tri, dep in zip(uv[f][ok], depth[ok]):
                tris.append((dep, n, tri))
        tris.sort(key=lambda x: -x[0])
        out = np.zeros((size[1], size[0]), np.uint8)
        for _, n, tri in tris:
            cv2.fillConvexPoly(out, np.round(tri*4).astype(np.int32), n, lineType=cv2.LINE_8, shift=2)
        return out

    def fit_loss(self, frame):
        d = np.load(self.root/'processed_data/smpl'/f'{frame:05d}.npy', allow_pickle=True).item()
        return {w: float(d[w].get('epoch_loss', 0)) for w in PEOPLE}


def touching(gt, dil=7):
    a, b = gt == 1, gt == 2
    if not a.any() or not b.any():
        return False
    k = np.ones((dil, dil), np.uint8)
    return bool((cv2.dilate(a.astype(np.uint8), k) & b).any())
