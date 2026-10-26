"""Exact installed processor coordinate fixture; no model weights or inference."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import atomic_json,restore_grid,sha256


def audit(model_dir):
    import transformers
    from transformers import AutoImageProcessor
    import transformers.models.qwen3_5.modeling_qwen3_5 as model_source
    processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True)
    patch=int(processor.patch_size);merge=int(processor.merge_size)
    records=[]
    for gh,gw in ((4,6),(8,4),(160,90)):
        # Values identify every row/column independently, not just a uniform image.
        yy,xx=np.meshgrid(np.arange(gh,dtype=np.uint8),np.arange(gw,dtype=np.uint8),indexing='ij')
        im=np.zeros((gh*patch,gw*patch,3),np.uint8)
        im[...,0]=np.repeat(np.repeat(yy,patch,0),patch,1)
        im[...,1]=np.repeat(np.repeat(xx,patch,0),patch,1)
        inputs=processor(images=Image.fromarray(im),do_resize=False,do_rescale=False,do_normalize=False,return_tensors='pt')
        temporal=int(processor.temporal_patch_size)
        tokens=inputs['pixel_values'][:,[0,temporal*patch*patch]].numpy()
        expected=np.stack([yy,xx],-1)
        corrected=restore_grid(tokens,gh,gw,merge)
        legacy=tokens.reshape(gh,gw,2)
        if not np.array_equal(corrected,expected):raise ValueError('Exact processor does not match proposed coordinate map')
        records.append({'grid_hw':[gh,gw],'corrected_mismatched_locations':int(np.any(corrected!=expected,axis=-1).sum()),'legacy_mismatched_locations':int(np.any(legacy!=expected,axis=-1).sum()),'total_locations':gh*gw})
    return {'transformers_version':transformers.__version__,'processor_class':type(processor).__name__,
            'patch_size':patch,'merge_size':merge,'fixtures':records,
            'processor_source_sha256':sha256(inspect.getfile(type(processor))),
            'model_source_sha256':sha256(inspect.getfile(model_source)),
            'model_config_sha256':sha256(Path(model_dir)/'config.json'),
            'conclusion':'Raw vision tokens use grouped spatial order. Direct reshape is not image order. Merged tokens require a separate contract; this fixture does not measure decoder-performance impact.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=audit(a.model_dir);atomic_json(a.out,r);print(json.dumps(r))
