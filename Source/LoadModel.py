from tensorflow.keras.models import load_model
import numpy as np
from Metrics import *
from Losses import *


def mga_load_model(model_path,fp_weight = 5.0):
	weighted_loss=mga_get_weighted_loss(np.array([[1.,fp_weight]]))
	co = {"weighted_loss":weighted_loss,"f1":f1,"precision":precision,"recall":recall}

	model = load_model(model_path,custom_objects=co)
	return model