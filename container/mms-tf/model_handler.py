import logging
import tensorflow as tf
import time
import os
import json

logger = logging.getLogger(__name__)


class AdsTFModelHandler(object):

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        tf.enable_eager_execution()
        self.model = tf.keras.models.load_model(model_dir)

    def inference(self, model_input):
        inp = json.loads(model_input[0].get("body"))
        feat = tf.constant(inp, dtype=tf.float32)
        response = self.model(feat, training=False).numpy()
        return [response.tolist()]

_service = AdsTFModelHandler()


def handle(data, context):
    #start_time = time.time()
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    response = _service.inference(data)
    #logger.info('TF MODEL PREDICT: {}'.format((time.time() - start_time) * 1000))
    return response
