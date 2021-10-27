# spacebridge
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

## Setup development environment

```sh
# configure dev environment
ploomber install


# ...or use conda directly
conda env create --file environment.yml # replace conda with mamba to use mamba

# activate conda environment
conda activate spacebridge
```



## Running the pipeline

```sh
ploomber build

# start an interactive session
ploomber interact
```

## Exporting to other systems

[soopervisor](https://soopervisor.readthedocs.io/) allows you to run ploomber projects in other environments (Kubernetes, AWS Batch, AWS Lambda and Airflow). Check out the docs to learn more.
