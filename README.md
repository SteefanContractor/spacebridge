# spacebridge
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

## Setup development environment

For all scripts except for train_pytorch_tabnet.py setup the conda/mamba environment with environment.pycaret.yml otherwise use environment.pytorch.yml. This is because of dependency conflicts between pycaret and pytorch.

```sh
# use conda directly
conda env create --file environment.pycaret.yml # replace pycaret with pytorch to setup pytorch environment
# use the conda -p/--prefix flag to specify install location e.g.
conda env create -f environment.pycaret.yml -p /path/to/envs/pycaret

# activate conda environment
conda activate pycaret
```

## UMAP

use the spacebridge-tabnet environment for correct umap pickle loading

## Running the pipeline

See [ploomber documentation](https://docs.ploomber.io/en/stable/index.html) to build using ploomber or right-click the .py files and select "open as notebook" to run the py files as a notebook.

```sh
ploomber build

# for building partial pipelines
ploomber build --partial [task_name]
# note this would also build all upstream tasks so set upstream to None in the script if you don't want that

# start an interactive session
ploomber interact
```

## Exporting to other systems

[soopervisor](https://soopervisor.readthedocs.io/) allows you to run ploomber projects in other environments (Kubernetes, AWS Batch, AWS Lambda and Airflow). Check out the docs to learn more.
