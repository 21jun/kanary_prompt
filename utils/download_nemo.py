from nemo.collections.asr.models import EncDecMultiTaskModel
import torch
from pathlib import Path
import tarfile
import gc

map_location = 'cpu'
MODEL_LIST = [
    # 'nvidia/canary-180m-flash',
    'nvidia/canary-1b-flash',
    # 'nvidia/canary-1b',
    # 'nvidia/canary-1b-v2',
]

for model_id in MODEL_LIST:
    name = model_id.split('/')[-1]
    out_dir = Path(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    nemo_path = out_dir / f"{name}.nemo"

    model = EncDecMultiTaskModel.from_pretrained(model_id, map_location=map_location)
    model.save_to(str(nemo_path))
    del model
    gc.collect()

    with tarfile.open(nemo_path, mode='r:*') as tf:
        tf.extractall(path=out_dir)
