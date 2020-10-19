import logging
import tvm
from tvm.contrib import graph_runtime
import time
import os
import io
import numpy as np

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
        loaded_graph = open(os.path.join(model_dir, "compiled_model.json")).read()
        loaded_lib = tvm.runtime.load_module(os.path.join(model_dir, "compiled.so"))
        loaded_params = bytearray(open(os.path.join(model_dir, "compiled.params"), "rb").read())
        self.model = graph_runtime.create(loaded_graph, loaded_lib, tvm.gpu())
        self.model.load_params(loaded_params)

    def inference(self, model_input):
        inp = np.load(io.BytesIO(model_input[0].get("body")), allow_pickle=True)
        self.model.set_input(0, inp)
        self.model.run()
        response = self.model.get_output(0).asnumpy()
        return [response.tolist()]

_service = AdsTFModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.inference(data)
