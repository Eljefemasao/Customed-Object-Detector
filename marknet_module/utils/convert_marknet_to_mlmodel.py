import keras
import sys
sys.path.append('../')
import os
from coremltools.proto import NeuralNetwork_pb2
from keras.models import load_model
import coremltools
import tfcoreml
from model.MarkNet_keras import Model_

image_shape = (100, 100, 3)
# MarkNetの呼び出し
net=Model_(input_shape=image_shape)
model=net.MarkNet()

model.load_weights('../trained_weights/best_model.hdf5', by_name=True)
#model = load_model('../model/plate.h5', custom_objects={'Normalize': convert_lambda})

model.author = 'm.matsunaga'
model.short_description = 'urasoe museum plate recognition'
#model.input_description['image'] = 'take a input an image of a plate'
#model.output_description['output'] = 'prediction of plate'
output_labels = ['0','1','2','3','4']

mlconverted_model = coremltools.converters.keras.convert(
      model,
      input_names="image",
      image_input_names="image",
      output_names="output")

mlconverted_model.save('plate.mlmodel')
