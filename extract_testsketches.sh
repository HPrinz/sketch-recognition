#!/usr/bin/env bash

wget  -P img-all/ "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
unzip img-all/sketches_png.zip

echo "downloaded sketches into folder img/"


for category in */ ; do
    echo "extracting for category $category"
    mkdir ../test/${category}

    i=0
    for x in ${category}/*; do
      if [ "$i" = 10 ]; then break; fi
      mv -- "${x}" ../test/${category}
      i=$((i+1))
    done


done
