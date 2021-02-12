import logging
import mxnet as mx
from mxnet.contrib import neuron
import time
import os
import io
import numpy as np
from collections import namedtuple

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

        n_data = 11
        inputs_list = []
        for i in range(n_data):
            inputs_list.append("data{}".format(i))

        self.batch_size = 50
        data_shape = [
                ('data0', (self.batch_size, 23)),
                ('data1', (self.batch_size, 1)),
                ('data2', (self.batch_size, 1)),
                ('data3', (self.batch_size, 1)),
                ('data4', (self.batch_size, 1)),
                ('data5', (self.batch_size, 15)),
                ('data6', (self.batch_size, 15)),
                ('data7', (self.batch_size, 15)),
                ('data8', (self.batch_size, 15)),
                ('data9', (self.batch_size, 15)),
                ('data10', (self.batch_size, 15))]
        self.mxnet_ctx = mx.neuron()
        load_symbol, args, auxs = mx.model.load_checkpoint(os.path.join(model_dir, "compiled"), 0)
        self.mod = mx.mod.Module(symbol=load_symbol, context=mx.neuron(), data_names=inputs_list, label_names=None)
        self.mod.bind(for_training=False, data_shapes=data_shape, label_shapes=self.mod._label_shapes)
        self.mod.set_params(args, auxs)

    def inference(self, model_input):
        inp = np.frombuffer(model_input[0].get("body"), dtype=np.float32)
        inp = np.reshape(inp, (self.batch_size, 117))
        Batch = namedtuple('Batch', ['data'])
        inputs_list=[]

        inputs_list.append(mx.nd.array(inp[:,0:23]))
        for i in range(1,5):
            inputs_list.append(mx.nd.array(inp[:,[i]]))

        for j in range(5,11):
            inputs_list.append(mx.nd.array(inp[:,i:i+15]))
            i+=15

        self.mod.forward(Batch(inputs_list))
        response = self.mod.get_outputs()[0].asnumpy()

        return [response.tolist()]

_service = AdsTFModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
