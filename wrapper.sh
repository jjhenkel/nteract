#!/bin/bash

mkdir -p "${3#-D}"

COMPILED="${1}"

shift

LD_LIBRARY_PATH=/usr/local/lib "${COMPILED}" $@ 
