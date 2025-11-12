#!/bin/bash

# Install the package
$PYTHON -m pip install . -vv --no-deps --no-build-isolation

# Make qmafft executable
chmod +x $SP_DIR/famus/qmafft

# Create models directory structure
mkdir -p $PREFIX/share/famus/models/full
mkdir -p $PREFIX/share/famus/models/light