#!/usr/bin/bash
set -e

# set up the FARSI submodule
cd Project_FARSI
git clone https://github.com/zaddan/cacti_for_FARSI
cd cacti_for_FARSI
make -j4
cd ../..
./apply_diffs.sh

# set up reference repo to create diffs against
git clone https://github.com/facebookresearch/Project_FARSI Project_FARSI_orig
cd Project_FARSI_orig
git clone https://github.com/zaddan/cacti_for_FARSI
rm -rf .git cacti_for_FARSI/.git
cd ..
chmod -R a-w Project_FARSI_orig
chmod -R ugo+r Project_FARSI_orig

# prepare the conda environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS"
    conda env create -f environment_linux.yml
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    conda env create -f environment_macos.yml
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi
conda activate artemis
