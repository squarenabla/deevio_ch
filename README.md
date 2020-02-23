Repository for solving the deevio challenge.  
train.py contains a script to train a classifier. 

Default parameters to be customized: \
MODEL_PATH = './model/1' \
DATA_PATH = './data/nailgun/nailgun/' 

predict.py contains a flask.app, which loads the model
and answers the requests according to the task.