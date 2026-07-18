#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Step 1: Query iNaturalist for family + species metadata ==="
python inaturalistAPItest.py query --taxon-file Taxon_ids/taxon_families.txt --pages 3

echo "=== Step 2: Download images (family + species folders) ==="
python inaturalistAPItest.py download --layout hierarchical

echo "=== Step 3: Train family classifier ==="
python model/model_training.py --data-dir data/families --checkpoint model/checkpoints/family_classifier.pth --epochs 10

echo "=== Step 4: Train per-family species classifiers ==="
python model/train_species_models.py --epochs 8

echo "=== Done ==="
echo "Run: streamlit run app.py"
