The scripts for the first approach requires the structure of the dataset to be as computed by the script in the data/ folder.


## main.sh

This script retrains for each attributes the inceptionv3 model. It calls retrain.py

## retrain.py 

Is a Tensorflow script that retrains the specified model. It has been slightly modified to test on the test dataset and not on a portion of the training data.