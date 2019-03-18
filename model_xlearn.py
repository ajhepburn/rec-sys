from os.path import join

import xlearn as xl

# Training task
rpath = './data/libsvm/'
modelpath = './models/xlearn/'

train, test = join(rpath, 'train.libsvm'), join(rpath, 'test.libsvm')

fm_model = xl.create_fm()                # Use field-aware factorization machine (ffm)
fm_model.setTrain(train)    # Set the path of training dataset
fm_model.setValidate(test)  # Set the path of validation dataset

# Parameters:
#  0. task: binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'auc'}

model_out = join(modelpath, 'model.out')
# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, model_out)

# Prediction task
fm_model.setTest(test)  # Set the path of test dataset
fm_model.setSigmoid()                 # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
fm_model.predict(model_out, join(rpath,'output.txt'))