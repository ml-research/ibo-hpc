# Hyperparameter Optimization via Interacting with Probabilistic Circuits
This repository contains the code of `Hyperparameter Optimization via Interacting with Probabilistic Circuits`.

## Abstract
Despite the growing interest in designing truly interactive hyperparameter optimization (HPO) methods, only a few allow human feedback to be included. However, these methods add friction to the interactive process, rigidly requiring users to define prior distributions ex ante and often imposing additional constraints on the optimization framework. This hinders flexible incorporation of expertise and valuable knowledge of domain experts, who might provide partial feedback at any time during optimization. To overcome these limitations, we introduce a novel Bayesian optimization approach leveraging probabilistic circuits (PCs) as a surrogate model. PCs encode a tractable joint distribution over the hybrid hyperparameter space and evaluation scores. They enable tractable and exact conditional inference and sampling, allowing users to provide beliefs interactively and generate configurations adhering to their feedback. We demonstrate the benefits of the resulting interactive HPO through an extensive empirical evaluation of diverse benchmarks, including the challenging setting of neural architecture search.

## Setup
We recommend to run the code in a `Docker` container, we don't guarantee a working version without Docker.

To setup the code and run experiments do the following instructions:

1. Navigate to the root directory of the repository
2. Run `docker build --build-arg baseimage=[BASE_IMAGE] --no-cache --tag ihpo .` where `BASE_IMAGE=python:3.9-slim` depending on whether you'd like to use GPU or not.
3. Run `docker run -v .:/app/ -it ihpo` to enter the Docker container.
4. Run `python main.py --exp nas101 --optimizer pc` to run the code (change options accordingly).

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

## Plotting
To plot the results of the optimization runs, swith to the directory `ihpo/utils`. Here you can execute `python plot.py` with several arguments (described in the argument definitions). Generating some plots can take a while (the regret\_vs\_cost up to 5-10 minutes). 

## Troubleshooting
Sometimes you might encounter an error of scikit-learn stating that some metric(s) cannot be imported.
There is a hacky workaround that resolves this issue:

1. `pip uninstall scikit-learn pandas`
2. `pip install scikit-learn pandas`
3. `pip uninstall scikit-learn`
4. `pip install scikit-learn==1.1.3`

This forces pip to downgrade to `scikit-learn==1.1.3.` while keeping the other packages at higher versions (might cause warnings by pip). The code should work after that.

If the automatic setup of the dataset fails, try to manually run the following command in your docker container:

`gdown 1KSjP3-5x1O0_2-DuQJgcXG3_h_AiYh0j --folder -O /app/data/`

If it does not succeed, try to download the contents of the Google Drive directory with the given ID (`1KSjP3-5x1O0_2-DuQJgcXG3_h_AiYh0j`) manually one by one and place it in the `/app/data/` directory.

## Citation
If you find this code useful in your research, please consider citing:


    @incollection{seng2024tpm,
      title = {Hyperparameter Optimization via Interacting with Probabilistic Circuits},
      author = {Seng, Jonas and Ventola, Fabrizio and Yu, Zhongjie and Kersting, Kristian},
      booktitle = {7th Workshop on Tractable Probabilistic Modeling at UAI (TPM 2024)},
      year = {2024}
    }


## Acknowledgments
This work was supported by the National High-Performance Computing Project for Computational Engineering Sciences (NHR4CES) and the Federal Ministry of Education and Research (BMBF) Competence Center for AI and Labour ("KompAKI", FKZ 02L19C150). Furthermore, this work benefited from the cluster project "The Third Wave of AI".
