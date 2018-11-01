#!/usr/bin/env bash

cd ~/Nitin/ncsdk/Nitin/SpectralCompression/ &&
python TrainIdentity.py &&
python TestIdentity.py &&

cd Model/ &&
# Run on NCS:
mvNCCompile inference.meta -s 12 -in Input -on final/BiasAdd -o inference.graph &&

# Create HTML File (Profiling)
mvNCProfile inference.meta -s 12 -in Input -on final/BiasAdd -is 32 32 &&

# Checking
mvNCCheck inference.meta -s 12 -in=Input -on=final/BiasAdd -is 32 32 -cs 0,1,2
