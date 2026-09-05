#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/translation"
PYTHON_BIN="${PYTHON_BIN:-python}"
NAME="${NAME:-deep_drum_pix2pix}"
RUN="$ROOT/translation/checkpoints/$NAME"
DATA_ROOT="${DATA_ROOT:-$ROOT/data_construction/output/dataset/fold_AB}"
shopt -s nullglob
DATA_FILES=("$DATA_ROOT/train/"*.png)
DATASET_SIZE=${#DATA_FILES[@]}
if (( DATASET_SIZE == 0 )); then
    echo "No paired PNG files found in $DATA_ROOT/train" >&2
    exit 1
fi
mkdir -p "$RUN"
if [[ "${1:-}" != "--test-only" && -e "$RUN/latest_net_G.pth" ]]; then
    echo 'Existing weights found; refusing to overwrite this training run.' >&2
    exit 1
fi
exec >> "$RUN/pipeline.log" 2>&1
trap 'echo "FAILED $(date -Is)" > "$RUN/status.txt"' ERR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$SOURCE"
COMMON=(--dataroot "$DATA_ROOT"
    --name "$NAME" --checkpoints_dir "$ROOT/translation/checkpoints"
    --model pix2pix --direction AtoB --dataset_mode aligned
    --netG unet_256 --norm batch --gpu_ids 0)
if [[ "${1:-}" != "--test-only" ]]; then
echo "TRAINING $(date -Is)" > "$RUN/status.txt"
"$PYTHON_BIN" train.py "${COMMON[@]}" --batch_size 8 --num_threads 4 \
    --load_size 286 --crop_size 256 --n_epochs 100 --n_epochs_decay 100 \
    --lr 0.0002 --beta1 0.5 --lambda_L1 100 --gan_mode vanilla \
    --display_id -1 --no_html --print_freq 400 \
    --save_epoch_freq 5 --save_latest_freq 5000
cp "$RUN/train_opt.txt" "$RUN/training_options.txt"
fi
echo "TESTING $(date -Is)" > "$RUN/status.txt"
"$PYTHON_BIN" test.py "${COMMON[@]}" --phase train --epoch latest \
    --load_size 256 --crop_size 256 --num_test "$DATASET_SIZE" --eval \
    --results_dir "$ROOT/results"
cp "$RUN/train_opt.txt" "$RUN/testing_options.txt"
if [[ -e "$RUN/training_options.txt" ]]; then
    cp "$RUN/training_options.txt" "$RUN/train_opt.txt"
fi
"$PYTHON_BIN" - "$ROOT/results/$NAME/train_latest" "$DATASET_SIZE" <<'PY'
import sys
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1])
expected_count = int(sys.argv[2])
for label in ('real_A', 'fake_B', 'real_B'):
    paths = list((root / 'images').glob(f'*_{label}.png'))
    assert len(paths) == expected_count, (label, len(paths), expected_count)
    for path in paths:
        with Image.open(path) as image:
            image.verify()
assert (root / 'index.html').is_file()
print(f'Verified {expected_count} predictions and reference images.')
PY
echo "COMPLETE $(date -Is)" > "$RUN/status.txt"
