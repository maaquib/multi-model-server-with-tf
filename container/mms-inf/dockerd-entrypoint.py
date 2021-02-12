import subprocess
import sys
import shlex
import os
from retrying import retry
from subprocess import CalledProcessError
from sagemaker_inference import model_server

def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)

@retry(stop_max_delay=1000 * 50,
       retry_on_exception=_retry_if_error)
def _start_mms():
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    os.environ['MMS_MAX_REQUEST_SIZE']='536870912'
    os.environ['MMS_MAX_RESPONSE_SIZE']='536870912'
    os.environ['MMS_DEFAULT_WORKERS_PER_MODEL']='16'
    os.environ['MMS_JOB_QUEUE_SIZE']='500'
    os.environ['NEURONCORE_GROUP_SIZES']='1'
    # os.environ['OMP_NUM_THREADS']='1'
    # os.environ['MXNET_USE_OPERATOR_TUNING']='1'
    model_server.start_model_server(handler_service='/home/model-server/model_handler.py:handle')

def main():
    if sys.argv[1] == 'serve':
        user_ncgs = os.environ.get('NEURONCORE_GROUP_SIZES')
        if user_ncgs is None:
            os.environ['NEURONCORE_GROUP_SIZES'] = "1"
        user_workers = os.environ.get('SAGEMAKER_MODEL_SERVER_WORKERS')
        if user_workers is None:
            num_host_cores = os.environ.get("NEURON_CORE_HOST_TOTAL")
            if num_host_cores is None:
                os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = "1"
            else:
                os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = num_host_cores
        print("NEURONCORE_GROUP_SIZES {}".format(os.environ.get('NEURONCORE_GROUP_SIZES')))
        print("SAGEMAKER_MODEL_SERVER_WORKERS {}".format(os.environ.get('SAGEMAKER_MODEL_SERVER_WORKERS')))
        _start_mms()
    else:
        subprocess.check_call(shlex.split(' '.join(sys.argv[1:])))

    # prevent docker exit
    subprocess.call(['tail', '-f', '/dev/null'])

main()

