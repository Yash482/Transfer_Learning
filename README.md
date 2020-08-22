# Transfer_Learning
Used Vgg16 model to extract features.

# Data Extraction
First we run data.py file to separate our images of different classes. 
This file get the images of different classes separatele and make 3 directories for further use.
One training data, one validation data and last test data. We use only 3000 images for traing the model and 1000 images for validation and testing separately.
Normally we use a lot more data for training CNN for classification but not with Transfer Learning.

# Dataset Prepared
After running data.py file, we got the 3 required folders. 
then we load the images and scale them and set the parameters.
Now our dataset is ready.

# Build Model
First form keras, Vgg16 model is included but not the FC layers.
Vgg166 model along with it weights is used only for extracting features, we don't train them.
The pre-trained Vgg16 model is frozen so that they don't get train. They are already trained on imagenet dataset for classifying images among 1000 classes.
We just use that earlier training where features are extracted.

After that we made our model and add vgg model to it. Our model contains only FC layers for using the features from vgg model and use them to classify.

