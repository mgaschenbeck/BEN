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

from tensorflow.keras import Model, metrics,layers
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


img_hr_input = layers.Input(shape=(120,120,4))
img_mr_input = layers.Input(shape=(60,60,6))
img_lr_input = layers.Input(shape=(20,20,2))

#Block 1a:
xa = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1a_conv1')(img_hr_input)
xa= layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1a_conv2')(xa)
xa = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1a_pool')(xa)

model_a = Model(inputs = img_hr_input,outputs=xa)


#Block 1b:
xb = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1b_conv1')(img_mr_input)
xb= layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1b_conv2')(xb)

model_b = Model(inputs = img_mr_input,outputs=xb)

xc = layers.Concatenate()([model_a.output,model_b.output])

# Block 2c
xc = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(xc)
xc = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(xc)
xc = layers.MaxPooling2D((3, 3), strides=(3, 3), name='block2_pool')(xc)

model_c = Model(inputs = [img_hr_input,img_mr_input],outputs=xc)



#Block 2d
xd = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2d_conv1')(img_lr_input)
xd= layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2d_conv2')(xd)
xd= layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2d_conv3')(xd)
model_d = Model(inputs=img_lr_input,outputs=xd)

##NOW MIX IN LAST LAYERS


# Block 3
xe = layers.Concatenate()([model_c.output,model_d.output])
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(xe)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
x = layers.Flatten()(x)

predictions = Dense(NUMBER_OF_CLASSES, activation='sigmoid', kernel_initializer='uniform', name='predictions')(x)

model = Model(inputs=[img_hr_input,img_mr_input,img_lr_input],outputs=predictions)

opt=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=opt,loss=mga_get_weighted_loss(np.array([[1.,20.]])),metrics=[f1,precision,recall,'binary_accuracy'])

model_name = "MRTEST_{}".format(int(time.time()))
callbacks = [
        #EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(factor=0.3, patience=2, min_lr=0.000001, verbose=1),
        #ModelCheckpoint(modelFile, verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir='logs/{}'.format(model_name),profile_batch=0,update_freq='batch',write_graph=False)
    ]

model.fit_generator(generator=tor,validation_data=vor,epochs = 20,steps_per_epoch=train_batches,validation_steps=validate_batches,use_multiprocessing=True,workers=15,callbacks=callbacks)
