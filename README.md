Repository for solving the deevio challenge.  
train.py contains a script to train a classifier. 

Default parameters to be customized: \
MODEL_PATH = './model/1' \
DATA_PATH = './data/nailgun/nailgun/' 

predict.py contains a flask.app, which loads the model
and answers the requests according to the task.

Results of the models:

| Architecture | Train Error | Valid Error|
| :----------- | -----------: | -----------: |
|Simple CNN (3 Conv Layers)| 1.0 | 0.5 |
|ResNet24| 0.6 | 0.5 |
|ResNet50 (Transfer learning)| 0.9 | 1.0 |
