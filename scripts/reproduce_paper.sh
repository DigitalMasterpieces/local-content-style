#!/usr/bin/env bash
# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0
#
# Re-render the paper's examples with the PyTorch hooks and compare against the expected renders.
#
#   scripts/reproduce_paper.sh [--skip-existing] [--offload]
#
# Manifests, inputs and expected renders are in paper/. Extra arguments are passed to
# run_manifest.py. Output: outputs/<manifest>/.../comparison.png, the mean difference to the
# expected renders, and a locality table per case.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

run() {
  echo "=== $1"
  python scripts/run_manifest.py "paper/manifest_$1.json" "${@:2}" 2>&1 \
    | grep -vE "Could not reach|kept in float32|warnings.warn|local_dir_use_symlinks|it/s\]|s/it\]|torchvision|Redirects"
}

run showcase_teaser_dualmask_gleizes_sq "$@"                                       # Fig. 1
run showcase_depth "$@"                                                            # Fig. 4 left
run showcase_subject_gleizes "$@"                                                  # Fig. 4 middle
run showcase_saliency "$@"                                                         # Fig. 4 right
run supplemental_s1 "$@"                                                           # Fig. S1
run supplemental_s2matrix_kandinsky --crop 338 108 778 548 --cols 5 --tile 220 "$@"  # Fig. S2
