#!/bin/bash

mkdir -p "${2#-D}"

LD_LIBRARY_PATH=/usr/local/lib /build/souffle/src/souffle $@
