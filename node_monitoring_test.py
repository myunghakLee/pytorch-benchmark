import os
import itertools
import gc
import logging
import time
import hashlib

import torch
from torchbenchmark import list_models
import torch.autograd.profiler as profiler


MAIN_DIR = os.path.split(os.path.abspath(__file__))[0]


def make_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename="test.log")
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger



logger = make_logger()

GRAFANA_URL = 'localhost:3000'
DASHBORAD_PATH_NODE = 'rYdddlPWk/node-exporter-full'
DASHBORAD_PATH_DCGM = 'Oxed_c6Wz/nvidia-dcgm-exporter-dashboard'

params_test_devices = ('cuda',)#('cpu', 'cuda')
params_test_workloads = ('eval',)#('eval', 'train')
# params_test_models = ('resnet50', 'BERT', 'dlrm', 'nvidia_deeprecommender') # [m.name for m in list_models()]
params_test_models = (
    'BERT_pytorch',
    'alexnet',
    'densenet121',
    'dlrm',
    'drq',
    'fastNLP',
    'nvidia_deeprecommender',
    'pytorch_stargan',
    'pytorch_struct',
    'resnet18',
    'resnet50',
    'resnext50_32x4d',
    'shufflenet_v2_x1_0',
    'soft_actor_critic',
    'squeezenet1_1',
    'timm_efficientnet',
    'timm_nfnet',
    'timm_regnet',
    'timm_resnest',
    'timm_vision_transformer',
    'timm_vovnet',
    'tts_angular',
    'vgg16',
    'yolov3'
)


# params_test[:][0] -> device, params_test[:][1] -> workload, params_test[:][2] -> model
params_test = itertools.product(params_test_devices, params_test_workloads, params_test_models) 


def profile_one_step(func, cuda, niter, nwarmup=3):
    for i in range(nwarmup):
        func() # default niter == 1, for warmup 1 is enough(memory allocation, etc.)

    # with profiler.profile(record_shapes=True, use_cuda = cuda) as prof:
    #     func(niter)
    func(niter)

    return #prof
    #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

results = []
torch.set_grad_enabled(False)

TIME_EXPERIMENT_START = time.time()
EXP_ID = hashlib.sha224(str(TIME_EXPERIMENT_START).encode()).hexdigest()

logger.info(f'Experiment {EXP_ID[:7]} started.')
for params in params_test:
    device, workload, model = params
    #print(model) # debug

    found = False
    for Model in list_models():
        if model.lower() in Model.name.lower():
            found = True
            break

    # build the model and get the task to profile
    m = Model(device = device)
    test = getattr(m, workload)

    time_mark = time.time()
    prof_data = profile_one_step(test, device == 'cuda', 100)
    time_mark_end = time.time()
    logger.info(f'Type: node-exporter,\tExperiment: {EXP_ID[:7]}, Model: {model}, Grafana URL: http://{GRAFANA_URL}/d/{DASHBORAD_PATH_NODE}?from={time_mark * 1000}&to={time_mark_end * 1000}')
    logger.info(f'Type: DCGM,\t\tExperiment: {EXP_ID[:7]}, Model: {model}, Grafana URL: http://{GRAFANA_URL}/d/{DASHBORAD_PATH_DCGM}?from={time_mark * 1000}&to={time_mark_end * 1000}')
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    #prof_data.export_chrome_trace(os.path.join(MAIN_DIR, 'trace_data', f'trace_{model}.json'))

    results.append((params, prof_data))


time_mark_end = time.time()
logger.info(f'Experiment {EXP_ID[:7]} ended. Grafana URL(node-exporter): http://{GRAFANA_URL}/d/{DASHBORAD_PATH_NODE}?from={TIME_EXPERIMENT_START * 1000}&to={time_mark_end * 1000}')
logger.info(f'Experiment {EXP_ID[:7]} ended. Grafana URL(DCGM): http://{GRAFANA_URL}/d/{DASHBORAD_PATH_DCGM}?from={TIME_EXPERIMENT_START * 1000}&to={time_mark_end * 1000}')

