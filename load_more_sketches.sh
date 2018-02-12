#!/usr/bin/env bash

mkdir img-all

wget  -P img-all/ "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
echo "Downloaded sketches into folder img-all/"
unzip img-all/sketches_png.zip
echo "Unzipped sketches. Follow instructions in Readme.md"

