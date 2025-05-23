# Interactive Hyperparameter Optimization via Probabilistic Circuits
This repository contains the code of `Towards Interactive Hyperparameter Optimization
with Probabilistic Circuits`.

## Abstract
Despite the growing interest in designing truly interactive hyperparameter optimization (HPO) methods, to date, only a few allow to include human feedback. Existing interactive Bayesian optimization (BO) methods incorporate human beliefs by weighting the acquisition function with a user-defined prior distribution. However, in light of the non-trivial inner optimization of the acquisition function prevalent in BO, such weighting schemes do not always accurately reflect given user beliefs. We introduce a novel BO approach leveraging tractable probabilistic models named probabilistic circuits (PCs) as a surrogate model. PCs encode a tractable joint distribution over the hybrid hyperparameter space and evaluation scores. They enable exact conditional inference and sampling. Based on conditional sampling, we construct a novel selection policy that enables an acquisition function-free generation of candidate points (thereby eliminating the need for an additional inner-loop optimization) and ensures that user beliefs are reflected accurately in the selection policy. We provide a theoretical analysis and an extensive empirical evaluation, demonstrating that our method achieves state-of-the-art performance in standard HPO and outperforms interactive BO baselines in interactive HPO.

## Setup
We recommend to run the code in a `Docker` container, we don't guarantee a working version without Docker.

To setup the code and run experiments do the following instructions:

1. Navigate to the root directory of the repository
2. Run `docker build --build-arg baseimage=python:3.9-slim --no-cache --tag ihpo .`
3. Run `docker run -v [HOST_PATH]:/app/ -it ihpo` to enter the Docker container.
4. Run `python main.py --exp [exp] --optimizer [optimizer]` to run the code (change options accordingly).

> Note: The build process can take 15-30 minutes, depending on your machine. Also, there will be quite a few **warnings** due to dependency incompatibilities which cannot be resolved currently. However, the codebase works as intended, only further extensions might be affected by dependency issues.

> Note: Once you start the container the first time, a setup script will be called. This loads all the benchmarks from various online sources and will download approximately 6GB of data. Depending on your internet connection, this can also take a while.

> Note: The JAHS benchmark requires relatively large RAM resources (>10GB). Thus, JAHS experiments might not be executable on machines with less memory.

For example, you can run the following command to run an experiment on the NAS-Bench-101 benchmark using a TPE BO optimizer from optuna:

`python main.py --exp nas101 --num-experiments 1 --optimizer optunabo --task cifar10 --log-dir ./data/nas101/`

This command will take approximately 10-20 minutes (depending on the machine) and produce a `.csv` logfile in the directory `/app/data/nas101/`.


Currently, the following experiments are implemented:

| Benchmark Name |   Option Name     | 5-Word Description                               | URL to Paper/GitHub                                                                                   |
|---------------|--------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| NAS-Bench-101 |  nas101, nas101_int      | First public architecture dataset for NAS research. | [Paper](https://arxiv.org/abs/1902.09635), [GitHub](https://github.com/google-research/nasbench)     |
| NAS-Bench-201 |  nas201, nas201_int      | Extends NAS-Bench-101 with different search space. | [Paper](https://arxiv.org/abs/2001.00326), [GitHub](https://github.com/D-X-Y/NAS-Bench-201)          |
| JAHS-Bench-201|  jahs, jahs_int      | Joint architecture and hyperparameter search benchmark. | [Paper](https://arxiv.org/abs/2206.10555), [GitHub](https://github.com/automl/jahs_bench_201)        |
| LCBench       |   lc, lc_int     | Benchmark for low-cost hyperparameter optimization. | [Paper](https://arxiv.org/abs/2006.13799), [GitHub](https://github.com/automl/LCBench)               |
| PD1           |   pd1, pd1_int     | Details not found in current sources.            | [Paper](https://www.jmlr.org/papers/volume25/23-0269/23-0269.pdf), [GitHub](https://github.com/google-research/hyperbo) |
| FCNet         |   fcnet, fcnet_int     | Fully connected neural network benchmark.        | [Paper](https://arxiv.org/abs/1905.04970), [GitHub](https://github.com/automl/nas_benchmarks)        |
| HPO-B         |    hpob, hpob_int    | Hyperparameter optimization benchmark suite.     | [Paper](https://arxiv.org/abs/2106.06257), [GitHub](https://github.com/automl/HPOBench)              |


The "int"-prefix stands for interactive. 

The following optimizers are implemented:
| Optimizer     | Option Name | URL to Paper/GitHub                                                                 | Note |
|--------------|------------|------------------------------------------------------------------------------------|------|
| IBO-HPC      | pc         | this GitHub                                                                      |      |
| Optuna       | optunabo   | [Website](https://optuna.org/), [Documentation](https://optuna.readthedocs.io/en/stable/) | implements "BO w/ TPE" in paper   |
| skopt        | skopt      | [GitHub](https://github.com/scikit-optimize/scikit-optimize)                      |  implements "BO w/ RF" in paper    |
| PiBO         | pibo       | [Paper](https://openreview.net/pdf?id=MMAeCXIa89)                                 |      |
| BOPrO        | bopro      | [Paper](https://arxiv.org/pdf/2006.14608)                                         |      |
| Local Search | ls        | -                                                                                  |      |
| SMAC         | smac       | [GitHub](https://github.com/automl/SMAC3)                                         |      |
| PriorBand    | -         | [Paper](https://arxiv.org/pdf/2306.12370)                                         |      |
| Random Search | rs        | -                                                                                  |      |


> Note: PriorBand is currently not supported in the Docker environment. However, it can be started on the host machine (which might trigger some errors due to missing 
dependencies). The reason is that it is currently incompatible with packages installed in the Docker container. Will be fixed in future releases.

## Interactions
To simulate user interactions we allow you to put json-files into the `interventions` directory. The json-files should have the following structure:

```
[
    {
        "type": "good" or "bad",
        "kind": "point" or "dist"
        "intervention": {YOUR INTERVENTION},
        "iteration" ITERATION TO TRIGGER INTERACTION
    },
    ...
]
```

You can put as many interactions as you like into the json-file. The interactions will be evaluated top down, i.e. the first appearing in the list first and so on.
The `iteration` field only holds information for the optimizer when to use the interaction but it has no effect on the order of how the interactions are triggered. This means that **interactions have to be ordered correctly in the list!** Otherwise this might cause errors or unexpected behavior.

The "type" is only meta-data and specifies whether an interaction is intended to be beneficial or harmful. The "kind" field defines whether the interaction is a point-interaction, i.e. the user fixes a certain hyperparameter to a certain value or whether the interaction is a distribution, i.e. the user specifies a distribution over possible interactions. The "intervention" field then specifies the exact interaction as a dictionary where {"hyperparameter": VALUE or {"dist:" ["uniform, cat, gauss"], "parameters": [parameters]}}" is the syntax. The "iteration" field specifies the iteration the interaction is applied. See the existing json-files in the `interventions`-directory for examples.

## Troubleshooting
Sometimes you might encounter an error of scikit-learn stating that some metric(s) cannot be imported.
There is a hacky workaround that resolves this issue:

1. `pip uninstall scikit-learn pandas`
2. `pip install scikit-learn pandas`
3. `pip uninstall scikit-learn`
4. `pip install scikit-learn==1.1.3`

This forces pip to downgrade to `scikit-learn==1.1.3.` while keeping the other packages at higher versions (might cause warnings by pip). The code should work after that.

# Plotting
To reproduce the plots from the paper, please follow the instructions below:
1. Download the raw log data from [this URL](https://figshare.com/ndownloader/files/53323751): `wget -O ./data.zip https://figshare.com/ndownloader/files/53323751`
2. If it does not exist, create the directory `/app/data`
3. Extract the raw data to `/app/data`: `unzip ./data.zip -d /app/data/`
4. Navigate to `/app/ihpo/utils`
5. Execute `python plot.py --exp [BENCHMARK] --out-file plot.pdf --with-std` where `BENCHMARK` is one of the benchmark options from the table above (column "Option Name") 

# License
To this repository, the CC BY-NC 4.0 license applies.

## Citation
If you find this code useful in your research, please consider citing:


    @InProceedings{seng2025ihpo,
      title = {Hyperparameter Optimization via Interacting with Probabilistic Circuits},
      author = {Jonas Seng and Fabrizio Ventola and Zhongjie Yu and Kristian Kersting},
      booktitle = {Proceedings of the Fourth International Conference on Automated Machine Learning},
      year = {2025}
    }


## Acknowledgments
This work was supported by the National High-Performance Computing Project for Computational Engineering Sciences (NHR4CES) and the Federal Ministry of Education and Research (BMBF) Competence Center for AI and Labour ("KompAKI", FKZ 02L19C150). Furthermore, this work benefited from the cluster project "The Third Wave of AI".
