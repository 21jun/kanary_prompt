# Copyright (c) 2025, NVIDIA CORPORATION.
# Copyright (c) 2025, CANARY PROJECT.
# Licensed under the Apache 2.0 license.
# This module mirrors `nemo.collections.common.prompts.canary2` so we can
# register a Kanary-specific prompt formatter independent of NeMo releases.

import torch
from lhotse import MonoCut
from lhotse.cut import Cut, MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.canary import BOOL_FALSE, BOOL_TRUE, PNC_FALSE, PNC_TRUE
from nemo.collections.common.prompts.canary2 import (
    map_manifest_values_to_special_tokens as nemo_map_manifest_values_to_special_tokens,
)
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY2_BOCTX,
    CANARY_BOS,
    CANARY_EOS,
    CANARY_SPECIAL_TOKENIZER,
    CanaryTokenizer,
)

ITN_TRUE = BOOL_TRUE | {"itn", "<|itn|>"}
ITN_FALSE = BOOL_FALSE | {"noitn", "<|noitn|>"}
ITN_DEFAULT = {"default", "undefined", "itn:undefined", "<|itn:undefined|>"}
ITN_SPECIAL_TOKENS = {"<|itn|>", "<|noitn|>", "<|itn:undefined|>"}
ITN_ALL = ITN_TRUE | ITN_FALSE | ITN_DEFAULT | ITN_SPECIAL_TOKENS
ITN_TRUE_NORMALIZED = {str(v).lower() for v in ITN_TRUE}
ITN_FALSE_NORMALIZED = {str(v).lower() for v in ITN_FALSE}
ITN_DEFAULT_NORMALIZED = {str(v).lower() for v in ITN_DEFAULT}
TIMESTAMP_TRUE = BOOL_TRUE | {"timestamp", "<|timestamp|>"}
TIMESTAMP_FALSE = BOOL_FALSE | {"notimestamp", "<|notimestamp|>"}
DIARIZE_TRUE = BOOL_TRUE | {"diarize", "<|diarize|>"}
DIARIZE_FALSE = BOOL_FALSE | {"nodiarize", "<|nodiarize|>"}
FOREIGN_TOKENS = ("<|foreign:ko|>", "<|foreign:en|>", "<|foreign:undefined|>")
FOREIGN_KO_NORMALIZED = {"ko", "kor", "korean", "foreign:ko", "foreign_ko", "<|foreign_ko|>", "<|foreign:ko|>"}
FOREIGN_EN_NORMALIZED = {"en", "eng", "english", "foreign:en", "foreign_en", "<|foreign_en|>", "<|foreign:en|>"}
FOREIGN_UNDEFINED_NORMALIZED = {
    "undefined",
    "unknown",
    "auto",
    "foreign:undefined",
    "foreign_undefined",
    "<|foreign_undefined|>",
    "<|foreign:undefined|>",
}


class KanaryPromptFormatter(PromptFormatter):
    """
    Local copy of Canary2 prompt formatter registered under a different name so we
    can experiment with prompt tweaks independent of Nemo releases.
    """

    NAME = "kanary"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}|emotion||source_lang||target_lang||pnc||itn||timestamp||diarize||foreign|",
            "slots": {
                "decodercontext": Modality.Text,
                "emotion": Modality.TextLiteral(
                    "<|emo:undefined|>", "<|emo:neutral|>", "<|emo:angry|>", "<|emo:happy|>", "<|emo:sad|>"
                ),
                "source_lang": Modality.Text,
                "target_lang": Modality.Text,
                "pnc": Modality.TextLiteral(*(PNC_TRUE | PNC_FALSE)),
                "itn": Modality.TextLiteral(*ITN_ALL),
                "timestamp": Modality.TextLiteral(*(TIMESTAMP_TRUE | TIMESTAMP_FALSE)),
                "diarize": Modality.TextLiteral(*(DIARIZE_TRUE | DIARIZE_FALSE)),
                "foreign": Modality.TextLiteral(*FOREIGN_TOKENS),
            },
        },
        "user_partial": {
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}",
            "slots": {"decodercontext": Modality.Text},
        },
        OUTPUT_ROLE: {
            "template": f"|text|{CANARY_EOS}",
            "slots": {"text": Modality.Text},
        },
    }

    def encode_dialog(self, turns: list[dict]) -> dict[str, torch.Tensor]:
        normalized_turns = []
        for turn in turns:
            normalized_turn = turn.copy()
            slots = turn.get("slots")
            if slots:
                normalized_turn["slots"] = slots.copy()
                if "foreign" in slots:
                    normalized_turn["slots"]["foreign"] = _normalize_foreign_slot(slots["foreign"])
            normalized_turns.append(normalized_turn)
        return super().encode_dialog(normalized_turns)

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        slot_values = map_manifest_values_to_special_tokens(slot_values)
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()

    itn_value = slot_values.pop("itn", None)
    foreign_value = slot_values.pop("foreign", None)
    slot_values = nemo_map_manifest_values_to_special_tokens(slot_values)

    needs_prompt_language = False
    prompt_language_missing = PromptFormatter.PROMPT_LANGUAGE_SLOT not in slot_values

    if itn_value is not None:
        slot_values["itn"] = _normalize_itn_slot(itn_value)
        needs_prompt_language |= slot_values["itn"] in ITN_SPECIAL_TOKENS

    if foreign_value is not None:
        slot_values["foreign"] = _normalize_foreign_slot(foreign_value)
        needs_prompt_language |= slot_values["foreign"] in FOREIGN_TOKENS

    if needs_prompt_language and prompt_language_missing:
        slot_values[PromptFormatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER
    return slot_values


def _normalize_foreign_slot(value: str) -> str:
    if value in FOREIGN_TOKENS:
        return value

    normalized = str(value).strip().lower()
    if normalized in FOREIGN_KO_NORMALIZED:
        return "<|foreign:ko|>"
    if normalized in FOREIGN_EN_NORMALIZED:
        return "<|foreign:en|>"
    if normalized in FOREIGN_UNDEFINED_NORMALIZED:
        return "<|foreign:undefined|>"
    raise ValueError(f"Unsupported foreign slot value '{value}'. Expected one of {FOREIGN_TOKENS} or their short forms.")


def _normalize_itn_slot(value: str) -> str:
    if value in ITN_SPECIAL_TOKENS:
        return value

    normalized = str(value).strip()
    normalized_lower = normalized.lower()
    if normalized_lower in ITN_TRUE_NORMALIZED:
        return "<|itn|>"
    if normalized_lower in ITN_FALSE_NORMALIZED:
        return "<|noitn|>"
    if normalized_lower in ITN_DEFAULT_NORMALIZED:
        return "<|itn:undefined|>"
    raise ValueError(
        f"Unsupported itn slot value '{value}'. Expected one of {ITN_SPECIAL_TOKENS} or their short forms."
    )


@registered_prompt_format_fn(Cut, KanaryPromptFormatter)
def kanary(cut: Cut, prompt: KanaryPromptFormatter) -> dict[str, torch.Tensor]:
    if isinstance(cut, MixedCut):
        cut = cut._first_non_padding_cut
    if not isinstance(cut, MonoCut):
        raise TypeError(
            f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
        )

    expected_slots = {"source_lang", "target_lang"}
    missing_keys = expected_slots - set(cut.custom)
    if missing_keys:
        raise RuntimeError(
            f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
            f"Please ensure that every utterance in the input manifests contains these keys."
        )

    optional_slots = {
        "decodercontext": "",
        "emotion": "<|emo:undefined|>",
        "itn": "<|itn:undefined|>",
        "timestamp": "<|notimestamp|>",
        "diarize": "<|nodiarize|>",
        "pnc": "<|pnc|>",
        "foreign": "<|foreign:undefined|>",
    }
    slots = {slot: cut.custom[slot] for slot in expected_slots}
    slots[prompt.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER
    for k, v in optional_slots.items():
        slots[k] = cut.custom[k] if k in cut.custom else v

    turns = [dict(role="user", slots=slots)]
    text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
    turns.append(
        dict(
            role="assistant",
            slots={
                "text": text,
                prompt.PROMPT_LANGUAGE_SLOT: ifnone(cut.supervisions[0].language, cut.custom.get("target_lang")),
            },
        ),
    )
    ans = prompt.encode_dialog(turns)
    if isinstance(prompt.tokenizer, CanaryTokenizer):
        eos = prompt.tokenizer.eos
    else:
        eos = prompt.tokenizer.token_to_id(CANARY_EOS)
    assert eos > -1, "Invalid tokenizer: tokenizer.token_to_id('{CANARY_EOS}') returned {eos}"
    assert (
        ans["answer_ids"][-1].item() == eos
    ), f"Expected the last token in answer_ids to be EOS, but we got {ans['answer_ids']}"
    ans["answer_ids"] = ans["answer_ids"][:-1]
    return ans


__all__ = ["KanaryPromptFormatter", "kanary"]
