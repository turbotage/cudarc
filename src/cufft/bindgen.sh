#!/bin/bash
set -exu

#ifolder=/usr/local/cuda/include
while getopts I: flag
do
  case "${flag}" in
    I) ifolder=${OPTARG};;
  esac
done

echo -I$ifolder

bindgen \
  --allowlist-type="^cufft.*" \
  --allowlist-function="^cufft.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I$ifolder \
  > sys.rs