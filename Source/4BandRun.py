from GeotiffGeneratorAllBands import *
from Metrics import *
from Losses import *

import numpy as np
import os
import time
import random
import tensorflow
import json
import pickle

from tensorflow.keras import Model, metrics
from tensorflow.keras.applications import ResNet50,VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint, TensorBoard




import numpy as np
from sklearn.model_selection import train_test_split


#PROJECT VARIABLES:
DATA_DIR='/mnt/tier1/maschenb/BigEarthNet-v1.0'
#Neet bands 4,3,2 for RGB

NUMBER_OF_CLASSES=43
#LABEL_INTS={"no_ship":0,"cargo":1,"dredging":2,"fishing":3,"passenger":4,"pleasure_craft":5,"sailing":6,"tanker":7,"tug":8}
LABEL_INTS={}


def fileList(source):
    matches = []
    labels = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('B04.tif')):
                prefix = os.path.join(root, filename)[0:-7]
                #R,G,B,Nir8 -all 10meter, then VegRedEdges(5,6,7,8A),SWIR11,SWIR12 - all 20 meter, then Coastal1,WaterVapour9
                #matches.append([prefix+'B04.tif',prefix+'B03.tif',prefix+'B02.tif',])
                matches.append([prefix+'B04.tif',prefix+'B03.tif',prefix+'B02.tif',prefix+'B08.tif',prefix+'B05.tif',prefix+'B06.tif',prefix+'B07.tif',prefix+'B8A.tif',prefix+'B11.tif',prefix+'B12.tif',prefix+'B01.tif',prefix+'B09.tif'])
                labels.append(os.path.join(root,os.path.basename(root)+'_labels_metadata.json'))

    return matches,labels



#File and label info:
train_ratio=.7
filenames,classlabels=fileList(DATA_DIR)
# count=0
# for classlabel_file in classlabels:
#     with open(classlabel_file) as f:w
#         cl_dict=json.load(f)
#         for l in cl_dict['labels']:
#             if l not in LABEL_INTS.keys():
#                 LABEL_INTS[l]=count
#                 count=count+1
with open("/mnt/panasas/maschenb/BigEarthNet/labels.json") as f:
    LABEL_INTS = json.load(f)



nSamples = len(filenames)
filenames_train,filenames_validate,labels_train,labels_validate = train_test_split(filenames,classlabels,test_size=(1.0-train_ratio))

filenames_dictionary = {"f_train":filenames_train,"l_train":labels_train,"f_validate":filenames_validate,"l_validate":labels_validate,"label_int":LABEL_INTS}
pickle.dump(filenames_dictionary,open( "files_split.p", "wb"))




# Parameters:
flow_params = { 'dim': (120,120),
                'batch_size': 64,
                'n_classes': NUMBER_OF_CLASSES,
                'shuffle': True}
train_batches=len(filenames_train)/flow_params['batch_size']/3
validate_batches=len(filenames_validate)/flow_params['batch_size']/3

augmentation_params={'horizontal_flip':True,'vertical_flip':True}#'rotation_range':90,'zoom_range':.2,"channel_shift_range":.1,'horizontal_flip':True,'vertical_flip':True}



# Generator - this is an object that specifies where and how we grab images
#Note, the ImageDataGenerator class was extended to add a flow_from_gernator command.  This has been customized
#to automatically get a class label from the directory name. Change if directory structure changes!
training_generator = ImageDataGeneratorExtended(filenames_train,labels_train,LABEL_INTS, **augmentation_params)
validation_generator = ImageDataGeneratorExtended(filenames_validate,labels_validate,LABEL_INTS, **augmentation_params)

#Iterator - this is the object we use, it iterates over the objects and gets the images with "flow param" sizes
tor=training_generator.flow_from_generator(**flow_params)
vor=validation_generator.flow_from_generator(**flow_params)

#Example of generator in use:
#X,y=tor.next()


#Example of model using generator:
init_model=VGG16(include_top=False,weights=None,input_shape=(120,120,4))
x = Flatten(name='flatten')(init_model.output)
predictions = Dense(NUMBER_OF_CLASSES, activation='sigmoid', kernel_initializer='uniform', name='predictions')(x)
model = Model(inputs = init_model.inputs[0], outputs = predictions)
for layer in init_model.layers:
    layer.trainable = True


opt=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)


model.compile(optimizer=opt,loss=mga_get_weighted_loss(np.array([[1.,5.]])),metrics=[f1,precision,recall,'binary_accuracy'])


model_name = "HR_FP5_{}".format(int(time.time()))
callbacks = [
        #EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(factor=0.3, patience=2, min_lr=0.000001, verbose=1),
        #ModelCheckpoint(modelFile, verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir='logs/{}'.format(model_name),profile_batch=0,update_freq='batch',write_graph=False)
    ]

model.fit_generator(generator=tor,validation_data=vor,epochs = 20,steps_per_epoch=train_batches,validation_steps=validate_batches,use_multiprocessing=True,workers=15,callbacks=callbacks)


