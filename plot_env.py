import os
import torch
import itertools
from torchbenchmark import list_models
import torch.autograd.profiler as profiler
import gc
from torch.autograd import DeviceType
import pandas as pd



MAIN_DIR = os.path.split(os.path.abspath(__file__))[0]


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

    with profiler.profile(record_shapes=True, use_cuda = cuda) as prof:
        func(niter)

    return prof
    #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

results = []
torch.set_grad_enabled(False)
for params in params_test:
    device, workload, model = params
    print(model) # debug

    found = False
    for Model in list_models():
        if model.lower() in Model.name.lower():
            found = True
            break

    # build the model and get the task to profile
    m = Model(device = device)
    test = getattr(m, workload)

    prof_data = profile_one_step(test, device == 'cuda', 100)
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    prof_data.export_chrome_trace(os.path.join(MAIN_DIR, 'trace_data', f'trace_{model}.json'))

    results.append((params, prof_data))



data_profiledtime_dict = {}
for result in results:
    events = result[1].function_events
    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])

    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            # in legacy profiler, kernel info is stored in cpu events
            if evt.is_legacy:
                sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == DeviceType.CUDA:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total

    data_profiledtime_dict[result[0]] = (sum_self_cpu_time_total, sum_self_cuda_time_total)


df = pd.DataFrame()
df['model'] = [r[0][2] for r in results]
df['time_cpu'] = [data_profiledtime_dict[r[0]][0] for r in results]
df['time_cuda'] = [data_profiledtime_dict[r[0]][1] for r in results]

EXPERIMENT_NAME = 'cuda_eval_52W_nograd'

df.to_pickle(EXPERIMENT_NAME + '.pickle')

