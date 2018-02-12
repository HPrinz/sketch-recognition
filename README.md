# Sketch Recognition

A SVM based machine learning program for human sketch recognition based on [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/).

<span>
<img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="100">
<img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="100">
<img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="100">
<img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="100">
<img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="100">
</span>

## Installation

- run `/install.sh`
- install [python 3 :snake:](https://www.python.org/downloads/)
- install [anaconda :snake:](https://conda.io/docs/user-guide/install/index.html)
- cd into project folder in terminal
```bash
conda env create -f anaconda/environment.yml
source activate sketch-recoginition
python3 train_sketchmodel.py
```

## Train more or different categories

The number of categories is currently set to 40 (see folders in `/img`)in order to reduce training time. if you want to include other or more categories, follow these steps: 

- run `load_more.sketches.sh` to download all sketches collected by [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) into folder `/img-all
- delete category folders in `/img`
- copy all categories you want to recognize into folder `/img
- run `./resize.sh` in folder `/img` as soon as you have finished copying
- run `./extract_testsketches.sh` in `/img` to move 10 sketches per categorey for testing
- run `train_sketchmodel.py` to train the SVM

## Train with Google QuickDraw Dataset

Google collected a massive amount of sketches in their QuickDraw Dataset which can be used to train the SVM as well:

- download a number of categories from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap into `/img/qd`
- rename files to `<category>.npy`
- run `train_sketchmodel_quickdraw.py` to train the SVM

## Using Trained Models

Trained models are saved in the `models/` folder for later use.
To reuse a pre-trained model, use `use_sketchmodel.py --modelname "models/file.sav"` (change model file accordingly)

## Results



## Credits and Thanks

The implementation of the SVM is based on the exercise code providede by [Prof. Dr.-Ing. Kristian Hildebrand](http://hildebrand.beuth-hochschule.de/#/)
The approach is described in [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)