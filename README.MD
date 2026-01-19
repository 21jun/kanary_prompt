
# Kanary Prompt

Lightweight package that registers the [`Kanary`](https://huggingface.co/lee1jun/kanay-1b-20260119) prompt formatter for NeMo/Canary2 pipelines.

## Install from Git

```
pip install git+https://github.com/21jun/kanary_prompt.git
```


## Local development install

```
pip install -e .
```

## Usage

```py
import nemo.collections.asr as nemo_asr
from kanary_prompt import kanary # by importing kanary, it will automatically register kanary prompt formatter to nemo lib.
import json
import tempfile
import soundfile as sf

# Initialize ASR model
asr_model = nemo_asr.models.ASRModel.from_pretrained("lee1jun/kanary-1b-20260119")

```

The formatter plugs into NeMo's prompt registry so you can reference `"kanary"` anywhere NeMo expects a prompt format name. Dependencies (`torch`, `lhotse`, `nemo_toolkit`) should already be present in your environment; install any GPU-specific variants per the PyTorch/NeMo docs.

## Build a wheel (optional)

```
python -m build
```
