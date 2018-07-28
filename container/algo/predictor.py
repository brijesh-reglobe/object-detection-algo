# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback


import flask

import pandas as pd


from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
import numpy as np
import tensorflow as tf

prefix = '/opt/ml/'
model_path = "model/resnet50_coco_best_v2.1.0.h5"

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None   # Where we keep the model when it's loaded
    graph = None
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if (cls.model == None) or (cls.graph == None):
            # download from s3 bucket
            cls.model = load_model(os.path.join(prefix, model_path), backbone_name='resnet50')
            cls.graph = tf.get_default_graph()
        return cls.model


    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        if (cls.model == None) or (cls.graph == None):
            clf = cls.get_model() 
        
        with cls.graph.as_default():
            clf = cls.get_model()
            result = clf.predict_on_batch(input)
        return result        

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    image = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'image/jpg':
        nparr = np.fromstring(flask.request.data, np.uint8)
        # decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #image = read_image_bgr('000000008021.jpg')
        image = preprocess_image(image)
        image, scale = resize_image(image)
        image = np.expand_dims(image, axis=0)
    else:
        return flask.Response(response='This predictor only supports Image data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    import json_tricks
    predictions = ScoringService.predict(image)
    result = {}
    boxes, scores, labels = predictions
    result["boxes"] = boxes
    result["scores"] = scores
    result["labels"] = labels

    return flask.Response(response=json_tricks.dumps(result), status=200, mimetype='text/csv')
