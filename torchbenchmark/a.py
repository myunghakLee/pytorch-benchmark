import pathlib


p = pathlib.Path("./").parent.joinpath("models")
_list_model_paths = sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())

import os
import importlib

def list_models(model_match=None):
    models = []
    for model_path in _list_model_paths:
        model_name = os.path.basename(model_path)
        print(model_name)
#         try:
#             module = importlib.import_module(f'.models.{model_name}', package=__name__)
#         except ModuleNotFoundError as e:
#             print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
#             continue
        try:
            module = importlib.import_module(f'.models.{model_name}', package=__name__)
        except ModuleNotFoundError as e:
            print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
            continue
        Model = getattr(module, 'Model', None)
#         if Model is None:
#             print(f"Warning: {module} does not define attribute Model, skip it")
#             continue
#         if not hasattr(Model, 'name'):
#             Model.name = model_name

#         # If given model_match, only return full or partial name matches in models.
#         if model_match is None:
#             models.append(Model)
#         else:
#             if model_match.lower() in Model.name.lower():
#                 models.append(Model)
#     return models

list_models()