The slim folder is a TensorFlow module that has been slightly modified to be adapted on our data set.

To run this approach, you should run one of the finetune_inception_v3_on_flowers_*.sh script that will download the pretrained model, extract the .tfrecord file from the datasets, retrain and evaluate.

## Precautions:
- Make sure you modify the global paths in the chosen script.
- Before running the script you also need to manually change the path in download_and_convert_data_MTFL.py line64 that is the path to the testing dataset.
- training dataset and testing dataset should have this structure:
	flower_photos
		-->attribute1
			--> *.jpg file
		-->attribute2
			--> *.jpg file
		...

The name of the folder "flower_photos" was not changed due to multiple dependancies.
- The name of the file created for the database will be flower_train*.tfrecord, flower_validation*.tfrecord or flower_testing*.tfrecord