# notebooks

## jupyter setup

local (repo-contained config):

```
mkdir -p .jupyter .jupyter_runtime .ipython
source .venv/bin/activate
IPYTHONDIR=./.ipython JUPYTER_DATA_DIR=./.jupyter JUPYTER_RUNTIME_DIR=./.jupyter_runtime jupyter lab --notebook-dir=.
```

kaggle:

```
uv venv .venv
UV_CACHE_DIR=/kaggle/working/.uv-cache uv pip install -r requirements.txt
./.venv/bin/jupyter lab --notebook-dir=/kaggle/working --ip=0.0.0.0 --port=8888 --no-browser
```

## notebooks

- `00_env_smoke.ipynb`: environment sanity checks.
- `10_data_prep.ipynb`: data layout validation and manifests.
- `20_route_a_train.ipynb`: dense kda hybrid experiments.
- `30_route_b_train.ipynb`: trm + kda experiments.
