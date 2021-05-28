# DL MINI-PROJECT: Classification, weight sharing, auxiliary losses

## Architecture

* `models.py` : contains the declaration of Net0, Net1 and Net2
* `advanced_models.py`: contains the declaration of LeNet, ResNet and ResNeXt
* `pre_processing.py`: contains functions that adapte the dataset extracted
from the prologue file so that it becomes a set of 14x14 images and not a set
of paires of images.
* `data_augmentation.py`: provide a data Augmenter that can be called with
different argument to augment the dataset given as argument.
* `processing.py`: contains most of the code used for the training of the 
models.
* `test.py`: contains code that train different models with different Loss.
The advanced models are commented because they take a little bit of time
to train. The data augmenter is commented also. To use it, simply comment
line 28 and un-comment line 27.