#!/bin/bash
# Pack the bundle + SLURM scripts into one archive to copy to the cluster.
# Run in WSL from the repo root after building the bundle:
#   ~/kraken-venv/bin/python cluster/bundle_for_cluster.py
#   bash cluster/make_tarball.sh
set -e
cd "$(dirname "$0")"          # -> cluster/
[ -d bundle ] || { echo "Run bundle_for_cluster.py first"; exit 1; }

tar czf edh_cluster.tar.gz \
    bundle \
    slurm_recognition.sh slurm_segmentation.sh setup_env.sh README.md

echo "Created cluster/edh_cluster.tar.gz ($(du -h edh_cluster.tar.gz | cut -f1))"
echo "Copy to cluster:  scp cluster/edh_cluster.tar.gz USER@CLUSTER:~/"
echo "On cluster (extracts into a clean folder, no wrapper inside the tar):"
echo "   mkdir -p edh && tar xzf edh_cluster.tar.gz -C edh && cd edh"
echo "   ls   # -> bundle/  slurm_recognition.sh  slurm_segmentation.sh  setup_env.sh  README.md"
