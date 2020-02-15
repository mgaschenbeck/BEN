import numpy as np
import gdal
import cv2
import os
import json
import tensorflow
import tensorflow as tf

class DataGenerator(tf.keras.preprocessing.image.Iterator):

    def __init__(self, image_data_generator,dim = (224,224),n_classes=8, batch_size=64,shuffle=True,seed=1234):
        
        #For augmentation
        self.image_data_generator = image_data_generator
        
        #For image format
        self.dim = dim
        self.n_classes = n_classes
        
        #For iterator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        #For the flow_from_generator method
        self.list_filenames = image_data_generator.list_filenames
        self.list_labelnames = image_data_generator.list_labelnames
        self.label_keys = image_data_generator.label_keys
        
        #call Iterator constructor:
        #__init__(n,batch_size,shuffle,seed)
        super().__init__(len(self.list_filenames), self.batch_size, self.seed ,self.shuffle)

    def _get_batches_of_transformed_samples(self, index_array):
        ''' Here you retrieve the images and apply the image augmentation, 
            then return the augmented image batch.

            index_array is just a list that takes care of the shuffling for you (see super class), 
            so this function is going to be called with index_array=[1, 6, 8] 
            if your batch size is 3
        '''
        current_batch_size=len(index_array)

        # Initialization
        #These are the 120x120 chips
        dim = np.array(self.dim,dtype='int')
        X1 = np.empty((current_batch_size, *dim, 4))
        #These are the 60x60 chips:
        X2 = np.empty((current_batch_size, *((dim/2).astype('int')),6))

        #Finally the 20x20 chips:
        X3 = np.empty((current_batch_size,*((dim/6).astype('int')),2))

        y = np.empty((current_batch_size, len(self.label_keys)), dtype=int)

        # Generate data
        list_filenames_temp = [self.list_filenames[i] for i in index_array]
        for i, filename in enumerate(list_filenames_temp):
            #High res
            ds=gdal.Open(filename[0])
            arr=ds.ReadAsArray()
            X1[i,:,:,0] = arr/arr.max()
            ds=gdal.Open(filename[1])
            arr=ds.ReadAsArray()
            X1[i,:,:,1] = arr/arr.max()
            ds=gdal.Open(filename[2])
            arr=ds.ReadAsArray()
            X1[i,:,:,2] = arr/arr.max()
            ds=gdal.Open(filename[3])
            arr=ds.ReadAsArray()
            X1[i,:,:,3] = arr/arr.max()
            
            #Medium res
            ds=gdal.Open(filename[4])
            arr=ds.ReadAsArray()
            X2[i,:,:,0] = arr/arr.max()
            ds=gdal.Open(filename[5])
            arr=ds.ReadAsArray()
            X2[i,:,:,1] = arr/arr.max()
            ds=gdal.Open(filename[6])
            arr=ds.ReadAsArray()
            X2[i,:,:,2] = arr/arr.max()
            ds=gdal.Open(filename[7])
            arr=ds.ReadAsArray()
            X2[i,:,:,3] = arr/arr.max()
            ds=gdal.Open(filename[8])
            arr=ds.ReadAsArray()
            X2[i,:,:,4] = arr/arr.max()
            ds=gdal.Open(filename[9])
            arr=ds.ReadAsArray()
            X2[i,:,:,5] = arr/arr.max()

            #Low res
            ds=gdal.Open(filename[10])
            arr=ds.ReadAsArray()
            X3[i,:,:,0] = arr/arr.max()
            ds=gdal.Open(filename[11])
            arr=ds.ReadAsArray()
            X3[i,:,:,1] = arr/arr.max()




            #X1[i,] = cv2.resize(X[i,],self.dim,interpolation=cv2.INTER_CUBIC)

            #Now transform
            #newparams = self.image_data_generator.get_random_transform(self.dim)
            #print(newparams)
            #X[i,]=self.image_data_generator.apply_transform(X[i,],newparams)


            # Store class
            with open(self.list_labelnames[index_array[i]]) as f:
                metaitem=json.load(f)
                clabels = metaitem['labels']
                

            y[i,]=self.make_one_hot(clabels)


        return [X1,X2,X3], y
    
    def make_one_hot(self,labelnamelist):
        vec = np.zeros(len(self.label_keys))
        for l in labelnamelist:
            vec[self.label_keys[l]]=1
        return vec



class ImageDataGeneratorExtended(tf.keras.preprocessing.image.ImageDataGenerator):
    '''
    This is our custom ImageDataGenerator.  ImageDataGenerator only flows from directory or numpy array.
    We need to use a generator to automatically figure out class labels from directory as well as
    to correctly load multispectral images - that's why we needd to define a flow_from_generator
    that returns our custom Iterator.
    '''
    def __init__(self,list_filenames,list_labelnames,label_keys,*args,**kwargs):
        self.list_filenames=list_filenames
        self.list_labelnames=list_labelnames
        self.label_keys=label_keys
        super().__init__(self,*args,**kwargs)


    def flow_from_generator(self, *args, **kwargs):
        return DataGenerator(self,*args, **kwargs)