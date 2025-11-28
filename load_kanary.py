# python build_kanary.py ~model.train_ds ~model.validation_ds
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from pathlib import Path

from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_EOS
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

from kanary_prompt import kanary
from kanary_prompt.formatter import KanaryPromptFormatter



if __name__ == '__main__':
    model_path = "/home/lee1jun/develop/kanary_prompt/kanary_models/kanary-1b-flash-agg.nemo"
    # model_path = "/home/lee1jun/develop/CANARY_PROJECT/canary-zoo/new_models/canary-1b-v2-ko-v5000_kanary.nemo"
    aed_model = EncDecMultiTaskModel.restore_from(model_path)

    print(aed_model)

    cfg = aed_model.cfg
    model_cfg = cfg.model if OmegaConf.select(cfg, "model") is not None else cfg
    prompt_format = OmegaConf.select(model_cfg, "prompt_format")
    prompt_defaults_cfg = OmegaConf.select(model_cfg, "prompt_defaults")
    prompt_defaults = (
        OmegaConf.to_container(prompt_defaults_cfg, resolve=True) if prompt_defaults_cfg is not None else None
    )

    rendered_prompt = None
    if prompt_defaults:
        user_defaults = next((item for item in prompt_defaults if item.get("role") == "user"), None)
        if user_defaults:
            rendered_prompt = KanaryPromptFormatter.TEMPLATE["user"]["template"]
            for slot, value in user_defaults.get("slots", {}).items():
                rendered_prompt = rendered_prompt.replace(f"|{slot}|", str(value))

    print(f"\nPrompt format: {prompt_format}")
    print(f"Prompt defaults:\n{OmegaConf.to_yaml(prompt_defaults_cfg) if prompt_defaults_cfg else 'None found'}")
    if rendered_prompt:
        print(f"Default user prompt string:\n{rendered_prompt}")

    # Tokenize an example with lang=ko and prompt using the special-token tokenizer for prompt pieces.
    example_text = "안녕하세요 반갑습니다"
    assistant_template = KanaryPromptFormatter.TEMPLATE[KanaryPromptFormatter.OUTPUT_ROLE]["template"]
    assistant_turn = assistant_template.replace("|text|", example_text)
    full_prompt_with_text = f"{rendered_prompt}{assistant_turn}" if rendered_prompt else assistant_turn

    tokenizer = aed_model.tokenizer
    
    prompt_tokens = []
    prompt_ids = []
    if rendered_prompt:
        prompt_tokens = tokenizer.text_to_tokens(rendered_prompt, lang_id="spl_tokens")
        prompt_ids = tokenizer.text_to_ids(rendered_prompt, lang_id="spl_tokens")

    text_tokens = tokenizer.text_to_tokens(example_text, lang_id="ko")
    text_ids = tokenizer.text_to_ids(example_text, lang_id="ko")
  


    combined_tokens = prompt_tokens + text_tokens if prompt_tokens else text_tokens
    combined_ids = prompt_ids + text_ids if prompt_ids else text_ids

    print("\n--- Tokenization example (lang_id=ko) ---")
    print(f"Full prompt + text:\n{full_prompt_with_text}")
    if prompt_tokens:
        print(f"Prompt tokens (spl_tokens):\n{prompt_tokens}")
        print(f"Prompt token IDs (spl_tokens):\n{prompt_ids}")
    print(f"Text tokens:\n{text_tokens}")
    print(f"Text token IDs:\n{text_ids}")
    print(f"Combined tokens (prompt + text):\n{combined_tokens}")
    print(f"Combined token IDs (prompt + text):\n{combined_ids}")