#!/usr/bin/env bash

# 1. Build Specialized Tokenizer (spl_tokens)

# python utils/build_kanary_special_tokenizer.py tokenizers/spl_tokens

# 2. Build Language Specific Tokenizers (Korean in this case)



# Build a comma-separated manifest argument from a multi-line list.
manifests=(
  /mnt/hdd/data/kspon_workspace/manifests/train_itn_nopnc_manifest.json
  /mnt/hdd/data/kspon_workspace/manifests/train_itn_pnc_manifest.json
  /mnt/hdd/data/kspon_workspace/manifests/train_noitn_nopnc_manifest.json
  /mnt/hdd/data/kspon_workspace/manifests/train_noitn_pnc_manifest.json
  /mnt/hdd/data/zeroth_workspace/manifests/train_zeroth_manifest_tagged.json
  /mnt/hdd/data/lecture_workspace/manifests/train_itn_noforeign_nopnc_manifest.json
  /mnt/hdd/data/lecture_workspace/manifests/train_itn_pnc_manifest.json
)
IFS=, MANIFESTS="${manifests[*]}"; unset IFS

MAXLEN=4
vocab_size=5000
NAME="ko"

echo "Building tokenizer with manifests: ${MANIFESTS}"

manifest_args=()
for manifest in "${manifests[@]}"; do
  manifest_args+=(--manifest "$manifest")
done

python utils/process_asr_text_tokenizer.py \
  "${manifest_args[@]}" \
  --vocab_size=${vocab_size} \
  --data_root=tokenizers/tokenizer_${NAME} \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --spe_max_sentencepiece_length=${MAXLEN} \


#   --tokens-file tokenizers/spl_tokens_kanary.txt 


# The CLI automatically injects 