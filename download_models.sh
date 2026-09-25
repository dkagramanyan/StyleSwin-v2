#!/bin/bash
# Pure-bash prefetch of every pretrained weight the combra metrics need, straight into
# the caches the libraries look in -- no Python, no GPU required. StyleSwin itself has
# no pretrained weights (a GAN trained from scratch), so only the combra InceptionV3 /
# CLIP / DINOv2 backbones are fetched. Run on a node WITH internet (e.g. a login node);
# the weights cache under $HOME, shared with the offline compute nodes, so the training
# jobs then need no network.
#
# URLs and on-disk filenames are pinned to what the installed libraries expect
# (pytorch-fid, open_clip 'openai' via the HuggingFace hub, torch.hub dinov2). If a
# library version differs it may still re-download at runtime -- verify the files below
# exist afterwards.
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (the defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh
# With MODEL_CACHE set, point the jobs at it:
#   export TORCH_HOME=$MODEL_CACHE/torch HF_HOME=$MODEL_CACHE/huggingface
set -u

MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
HUB_CKPT="${MODEL_CACHE}/torch/hub/checkpoints"   # pytorch-fid + dinov2 weights
HUB_DIR="${MODEL_CACHE}/torch/hub"                # torch.hub repo code (dinov2)
HF_HUB="${MODEL_CACHE}/huggingface/hub"           # HuggingFace hub cache (open_clip weights)
mkdir -p "$HUB_CKPT" "$HUB_DIR" "$HF_HUB"

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: need wget or curl on PATH." >&2
    exit 1
fi

status=0
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  exists: ${dest##*/}"
        return 0
    fi
    echo "  downloading: ${url##*/}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    else
        curl -fL -o "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    fi
}

echo "Caching pretrained models under: $MODEL_CACHE"

echo; echo "[1/3] InceptionV3 FID weights (combra fid) -> $HUB_CKPT"
fetch "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" "$HUB_CKPT/pt_inception-2015-12-05-6726825d.pth"

echo; echo "[2/3] CLIP ViT-L-14-336 'openai' (combra cmmd) -> $HF_HUB"
# open_clip loads the 'openai' tag from the HF repo timm/vit_large_patch14_clip_336.openai
# (huggingface_hub is one of its dependencies), so lay the file out as the HF hub cache
# does: blobs/<sha256>, snapshots/<rev>/<file> -> blob, refs/main = <rev>.
CLIP_REV="81e38efc4637de5023b10e75a7f9bd1c6fa6b010"
CLIP_SHA="fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0"
CLIP_REPO_DIR="$HF_HUB/models--timm--vit_large_patch14_clip_336.openai"
mkdir -p "$CLIP_REPO_DIR/blobs" "$CLIP_REPO_DIR/refs" "$CLIP_REPO_DIR/snapshots/$CLIP_REV"
fetch "https://huggingface.co/timm/vit_large_patch14_clip_336.openai/resolve/$CLIP_REV/open_clip_model.safetensors" "$CLIP_REPO_DIR/blobs/$CLIP_SHA" \
    && ln -sfn "../../blobs/$CLIP_SHA" "$CLIP_REPO_DIR/snapshots/$CLIP_REV/open_clip_model.safetensors" \
    && printf '%s' "$CLIP_REV" > "$CLIP_REPO_DIR/refs/main"

echo; echo "[3/3] DINOv2 dinov2_vitl14 (combra fd_dinov2) -> $HUB_DIR"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" "$HUB_CKPT/dinov2_vitl14_pretrain.pth"
# torch.hub also needs the dinov2 model code (it normally fetches the repo itself).
if [[ -d "$HUB_DIR/facebookresearch_dinov2_main" ]]; then
    echo "  exists: facebookresearch_dinov2_main/"
elif command -v git >/dev/null 2>&1; then
    echo "  cloning facebookresearch/dinov2"
    git clone --depth 1 https://github.com/facebookresearch/dinov2 "$HUB_DIR/facebookresearch_dinov2_main" \
        || { echo "  FAILED: git clone dinov2"; status=1; }
else
    echo "  SKIPPED dinov2 repo: git not on PATH"; status=1
fi

echo
if [[ $status -eq 0 ]]; then
    echo "Done. All weights cached under $MODEL_CACHE."
else
    echo "Some downloads failed (see above)."
fi
exit $status
