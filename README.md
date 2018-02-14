# Sketch Recognition

A SVM based machine learning program for human sketch recognition based on [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/).

<span> <img src="https://media.giphy.com/media/xThtaiAAht03JkRjFe/giphy.gif" width="160"> <img src="https://media.giphy.com/media/26DN1nteDhIcwwGOY/giphy.gif" width="160">   <img src="https://media.giphy.com/media/l1KcQRJkG8JGphV0k/giphy.gif" width="160"> <img src="https://media.giphy.com/media/xThta1euv6mW2uBFmg/giphy.gif" width="160"> <img src="https://media.giphy.com/media/xThtaoqt6cwyAVnUdy/giphy.gif" width="160">   </span>

## TODOs

- [x] rename img to train
- [ ] test img to train
- [ ] find best model
- [x] try CNN
- [ ] implement paper CNN
- [ ] CNN Quickdraw
- [ ] CNN Documentation

## Installation

- run `/install.sh`
- install [python 3 :snake:](https://www.python.org/downloads/)
- install [anaconda :snake:](https://conda.io/docs/user-guide/install/index.html)
- cd into project folder in terminal
```bash
conda env create -f anaconda/environment.yml
source activate sketch-recoginition
```

## Train SVM

### TU-Berlin Sketch Dataset

The number of categories is currently set to 40 (see folder names in `/tu-train`) in order to reduce training time (about 15 minutes).
Each category consists of 70 training sketches (located in `/tu-train`) and 10 testing sketches (located in `/tu-test`).
```bash
python train_sketchmodel.py
```

In order to **change or extend the categories**, follow these steps: 

- run `load_all_tu_sketches.sh` to download all sketches collected by [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) into folder `/img-all`
- delete folders in `/tu-train`
- copy all categories you want to train into `/tu-train`
- run `./resize.sh` in folder `/tu-train` as soon as you have finished copying. 10 sketches of each category will be moved for testing
- train the SVM (see above)

### Google QuickDraw Dataset

Google provides a massive amount of sketches through the QuickDraw Dataset which can be used to train the SVM as well. In this repository, 14 categories are used to train the SVM. 290 sketches of each category are are used to train and 10 are used to test the SVM. This decision was made due to the lower quality and lower resolution (28x28) of the dataset.

```bash
python train_sketchmodel_quickdraw.py
```

In order to **change or extend the categories**, follow these steps: 

- download a number of categories from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap into `/quickdraw-train`
- rename files to `<category>.npy`
- run `python reduce.py` in `/quickdraw-train` to extract and export the first 300 sketches (290 train and 10 test images). The original files in `/quickdraw-train` can be deleted afterwards
- train the SVM (see above)

## Train CNN

### CNN (Convolution Neural Network) Architecture

The Architecture is implemented as described in [Sketch-a-Net that Beats Humans (Yu et al. 2015)](https://arxiv.org/pdf/1501.07873.pdf)

|   | Input       | Filter Size | Filter Num | Stride | Padding |  Output |
|---|-------------|:-----------:|:----------:|:------:|:-------:|:-------:|
| 0 | Conv        |             |            |        |         | 225x225 |
| 1 | Conv(ReLU)  |    15x15    |     64     |    3   |    0    |  71x71  |
|   | MaxPool     |     3x3     |            |    2   |    0    |  35x35  |
| 2 | Conv(ReLU)  |     3x3     |     128    |    1   |    0    |  31x31  |
|   | MaxPool     |     3x3     |            |    2   |    0    |  15x15  |
| 3 | Conv(ReLU)  |     3x3     |     256    |    1   |    1    |  15x15  |
| 4 | Conv(ReLU)  |     3x3     |     256    |    1   |    1    |  15x15  |
| 5 | Conv(ReLU)  |     3x3     |     256    |    1   |    1    |  15x15  |
|   | MaxPool     |     3x3     |            |    2   |    0    |   7x7   |
| 6 | Conv(ReLU)  |     7x7     |     512    |    1   |    0    |   1x1   |
|   | Dropout     |             |            |        |         |   1x1   |
| 7 | Conv(ReLU)  |     1x1     |     512    |    1   |    0    |   1x1   |
|   | Dropout     |             |            |        |         |   1x1   |
| 8 | Conv (ReLU) |     1x1     |     250    |    1   |    0    |   1x1   |


### TU-Berlin Sketch Dataset

In a first attempt, the 1050 sketches with 150x150 pixel vectors were passed to the Fashion-network with a poor outcome (test loss: 7.3705, test accurracy: 0.0333):
![](md-images/cnn_tu_loss_1.png) ![](md-images/cnn_tu_accuracy_1.png)



### Google QuickDraw Dataset


## Using Pre-Trained Models

Trained models are saved in the `models/` folder for later use.
To reuse a pre-trained model, use `use_sketchmodel.py --modelname "models/file.sav"` (change model file accordingly)

## Results

### TU-Berlin Sketch Dataset

| Nr.   | Type | keypoints       | C                |    gamma            | Kernel  | score | best  |
|-------|------|-----------------|:----------------:|:-------------------:|:-------:|:-----:|:-----:|
| 1     | SVM   | 150x150x50     | 1, 10, 100, 1000 | -                   | linear  | 0.63  | same   |
| 2     | SVM   | 150x150x**30** | 1, 10, 100, 1000 | -                   | linear  | 0.67  | same   |
| **3** | SVM   | 150x150x30     | 10, 100, 1000    | .00001, .0001, .001 | **rbf** | 0.69  | gamma : 0.0001, C: 100   |

**Best Result : #3**
![Best Result SVM](md-images/svm_tu-3.png)


## Google QuickDraw

| Nr.   | Type | keypoints       | C                |    gamma            | Kernel  | score | best  |
|-------|------|-----------------|:----------------:|:-------------------:|:-------:|:-----:|:-----:|
| 1     | SVM   | 150x150x30     | 1, 10, 100, 1000 | -                   | linear  |  0.70 | gamma: 0.001, C: 10 |

**Best Result : #1**
![Best Result SVM](md-images/svm_quickdraw-1.png)

## Credits and Thanks

The implementation of the SVM is based on the exercise code providede by [Prof. Dr.-Ing. Kristian Hildebrand](http://hildebrand.beuth-hochschule.de/#/)

The SVM approach is described in [How Do Humans Sketch Objects? (Eitz et al. 2012)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)

The CNN approach is described in [Sketch-a-Net that Beats Humans (Yu et al. 2015)](https://arxiv.org/pdf/1501.07873.pdf)