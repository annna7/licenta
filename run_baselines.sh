#!/usr/bin/env bash

# Default values
DATA_DIR="data"
DATASET="swat"
WINDOW_SIZE=100
#BASELINES="ocsvm,iso,eis,lof,var"
#BASELINES='ocsvm'
#BASELINES='iso,eis'
BASELINES='iso'
OCSVM_KERNELS="linear,poly,rbf,tophat"
LOF_NEIGHBORS=20
EIS_MODELS=5
EIS_SUBSAMPLE=""
ARMA_P=5
VAR_LAG=5
NYSTROEM_COMPONENTS=1000
OCSVM_SUBSAMPLE=""
TOPHAT_RADIUS=""
RESULTS_DIR="results"
DOWNSAMPLE=5000   # ← Make this empty by default

usage() {
  echo "Usage: $0 [--data_dir DIR] [--dataset NAME] [--window_size N]"
  echo "          [--baselines LIST] [--ocsvm_kernels LIST] [--lof_neighbors N]"
  echo "          [--eis_models N] [--eis_subsample N] [--arma_p N] [--var_lag N]"
  echo "          [--nystroem_components N] [--ocsvm_subsample N] [--tophat_radius R]"
  echo "          [--results_dir DIR] [--downsample N]"
  exit 1
}

# Parse long-form arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"; shift 2;;
    --dataset)
      DATASET="$2"; shift 2;;
    --window_size)
      WINDOW_SIZE="$2"; shift 2;;
    --baselines)
      BASELINES="$2"; shift 2;;
    --ocsvm_kernels)
      OCSVM_KERNELS="$2"; shift 2;;
    --lof_neighbors)
      LOF_NEIGHBORS="$2"; shift 2;;
    --eis_models)
      EIS_MODELS="$2"; shift 2;;
    --eis_subsample)
      EIS_SUBSAMPLE="$2"; shift 2;;
    --arma_p)
      ARMA_P="$2"; shift 2;;
    --var_lag)
      VAR_LAG="$2"; shift 2;;
    --nystroem_components)
      NYSTROEM_COMPONENTS="$2"; shift 2;;
    --ocsvm_subsample)
      OCSVM_SUBSAMPLE="$2"; shift 2;;
    --tophat_radius)
      TOPHAT_RADIUS="$2"; shift 2;;
    --results_dir)
      RESULTS_DIR="$2"; shift 2;;
    --downsample)
      DOWNSAMPLE="$2"; shift 2;;   # Capture --downsample into the variable
    -*|--*)
      echo "Unknown option: $1" >&2; usage;;
    *)
      break;;
  esac
done

# Build Python invocation
CMD=(python baseline_main.py
  --data_dir "$DATA_DIR"
  --dataset "$DATASET"
  --window_size "$WINDOW_SIZE"
  --baselines "$BASELINES"
  --ocsvm_kernels "$OCSVM_KERNELS"
  --lof_neighbors "$LOF_NEIGHBORS"
  --eis_models "$EIS_MODELS"
)

# Only append optional flags if non‐empty
if [[ -n "$EIS_SUBSAMPLE" ]]; then
  CMD+=(--eis_subsample "$EIS_SUBSAMPLE")
fi

CMD+=(--arma_p "$ARMA_P" --var_lag "$VAR_LAG")

if [[ -n "$NYSTROEM_COMPONENTS" ]]; then
  CMD+=(--nystroem_components "$NYSTROEM_COMPONENTS")
fi

if [[ -n "$OCSVM_SUBSAMPLE" ]]; then
  CMD+=(--ocsvm_subsample "$OCSVM_SUBSAMPLE")
fi

if [[ -n "$TOPHAT_RADIUS" ]]; then
  CMD+=(--tophat_radius "$TOPHAT_RADIUS")
fi

if [[ -n "$DOWNSAMPLE" ]]; then
  CMD+=(--downsample "$DOWNSAMPLE")
fi

CMD+=(--results_dir "$RESULTS_DIR")

# Execute
"${CMD[@]}"
