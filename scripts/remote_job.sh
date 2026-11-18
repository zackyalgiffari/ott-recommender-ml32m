#!/usr/bin/env bash
# Runs ON the Colab VM, from /content/app (started by scripts/colab_train.sh).
#
#   remote_job.sh smoke                       sampled users, 1 short epoch, no upload
#   remote_job.sh full [package args] [-- train args]
#                                             full data, train, evaluate, package (+ private upload)
#   remote_job.sh ablations [names...]        ID-only, MovieLens-only and Amazon-only models
#                                             (or only the named ones)
set -euo pipefail
MODE="${1:-full}"
shift || true
export OTT_DATA_DIR="${OTT_DATA_DIR:-/content/data}"

echo "[setup] $(date -u +%FT%TZ) mode=$MODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
# Only installs what Colab lacks; the preinstalled CUDA torch stack is left untouched.
python -m pip install -q -r requirements.txt
python -c "import torch, transformers, polars; print('[setup] torch', torch.__version__, '| transformers', transformers.__version__, '| polars', polars.__version__, '| cuda', torch.cuda.is_available())"
echo "[setup] done"

demo() {  # $1 = model folder
  python -m src.recommend --model "$1" --domain movielens --k 10 \
    "The Matrix (1999)" "Toy Story (1995)" "Inception (2010)" \
    --new-title "Dune: Part Two (2024). Genres: Sci-Fi, Adventure. Paul Atreides unites with the Fremen to wage war against House Harkonnen." \
    --new-title "The Holiday Romance (2024). Genres: Comedy, Romance. Two strangers swap homes for Christmas and fall in love."
  python -m src.recommend --model "$1" --domain amazon --k 10 "Justified" "Vikings" "The Tomorrow War"
}

case "$MODE" in
  smoke)
    S=2000
    python -m src.prepare --sample-users "$S" --report results/data_report.json
    python -m src.embed_items --sample-users "$S"
    python -m src.train --sample-users "$S" --epochs 2 --max-steps-per-epoch 300 --val-users 2000 --log-every 50
    python -m src.evaluate --sample-users "$S"
    python -m src.package --sample-users "$S"
    demo outputs/hub
    ;;
  full)
    PKG=(); TRN=(); after=0
    for a in "$@"; do
      if [[ "$a" == "--" ]]; then after=1; elif (( after )); then TRN+=("$a"); else PKG+=("$a"); fi
    done
    python -m src.prepare
    python -m src.embed_items
    python -m src.train "${TRN[@]}"
    python -m src.evaluate
    python -m src.package "${PKG[@]}"
    demo outputs/hub
    ;;
  ablations)
    python -m src.prepare --report results/data_report_ablations.json
    python -m src.embed_items
    mkdir -p results/ablations
    run() {  # name, eval domains, train args...
      local name="$1" doms="$2"; shift 2
      echo "[ablation] $name"
      python -m src.train --output-dir "outputs/$name" --history-out "results/ablations/${name}_history.json" "$@"
      python -m src.evaluate --model "outputs/$name/best" --domains "$doms" --out "results/ablations/$name.json"
    }
    WHICH=("$@")
    (( ${#WHICH[@]} )) || WHICH=(id_only movielens_only amazon_only)
    for a in "${WHICH[@]}"; do
      case "$a" in
        id_only) run id_only movielens,amazon --no-content ;;
        movielens_only) run movielens_only movielens --domains movielens ;;
        amazon_only) run amazon_only amazon --domains amazon ;;
        *) echo "unknown ablation $a"; exit 2 ;;
      esac
    done
    python -m src.ablations
    ;;
  *)
    echo "unknown mode $MODE"; exit 2 ;;
esac
echo "[done] $(date -u +%FT%TZ)"
