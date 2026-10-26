"""Where the tangle-tracker code finds data, models and scratch space. Every path can be set with an environment
variable; the defaults follow the author's layout (one shared copy of datasets and weights under ~/Code/shared).

  LIMB_WORKSPACE  root of the auto-labeller workspace: evidence/, autolabel/, reviews/ (the private footage lives here;
                  it is not redistributed)
  LIMB_H4D        Harmony4D tree (MIT; https://huggingface.co/datasets/Jyun-Ting/Harmony4D), with 03_grappling2/
  LIMB_SAM31      SAM 3.1 multiplex checkpoint (sam3.1_multiplex.pt, SAM License)
  LIMB_S3D        SAM 3D Body code checkout
  LIMB_BMP        BBoxMaskPose v2 checkout (GPL-3.0; only imported at run time, not distributed here)
  LIMB_SCRATCH    scratch space for intermediate arrays (default: <tmp>/limb-entanglement)
"""
import os
import tempfile
from pathlib import Path

CODE = Path(os.environ.get('LIMB_CODE', Path.home()/'Code'))
WORKSPACE = Path(os.environ.get('LIMB_WORKSPACE', CODE/'kakashi-research/dataset'))
H4D = Path(os.environ.get('LIMB_H4D', CODE/'shared/datasets/harmony4d/benchmark-4cam'))
SAM31_CKPT = Path(os.environ.get('LIMB_SAM31', CODE/'shared/models/sam3.1/sam3.1_multiplex.pt'))
S3D = Path(os.environ.get('LIMB_S3D', CODE/'movement-research/repos/sam-3d-body'))
BMP = Path(os.environ.get('LIMB_BMP', CODE/'kakashi-research/external/BBoxMaskPose'))
SCRATCH = Path(os.environ.get('LIMB_SCRATCH', Path(tempfile.gettempdir())/'limb-entanglement'))
