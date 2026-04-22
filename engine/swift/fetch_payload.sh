#!/bin/bash
# Fetch sample protein structures (E. coli 70S ribosome pre/post translocation)
# from RCSB. Default destination is ./data; override with $PROTEIN_DIR.
set -e

DATA_DIR="${PROTEIN_DIR:-data/4V9D}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Fetching protein structures from RCSB into $DATA_DIR ..."

if [ ! -f "4V9D.cif.gz" ]; then
    echo "Downloading 4V9D ..."
    curl -# -O https://files.rcsb.org/download/4V9D.cif.gz
else
    echo "4V9D.cif.gz already present."
fi

if [ ! -f "4V9C.cif.gz" ]; then
    echo "Downloading 4V9C ..."
    curl -# -O https://files.rcsb.org/download/4V9C.cif.gz
else
    echo "4V9C.cif.gz already present."
fi

echo "Done."
