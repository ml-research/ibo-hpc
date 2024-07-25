# Interactive Hyperparameter Optimization via Probabilistic Circuits
This repository contains the code of `Towards Interactive Hyperparameter Optimization
with Probabilistic Circuits`.

## Abstract
Algorithm configuration plays a crucial role in several domains of computer science.In fact, an algorithm performs well only if its hyperparameters are properlyconfigured, generally resulting in a tedious process of manual tuning. Hyperparameteroptimization (HPO) aims to automatize this process by reframing it as an optimizationtask. Although numerous HPO methods have emerged recently, a critical limitationexists in their lack of interactivity, which disregards the expertise and invaluableknowledge of domain experts. To fill this gap, we present a novel Bayesianoptimization approach that enables and leverages the involvement of active usersthroughout the optimization procedure. To this aim, we exploit the tractable andflexible inference of probabilistic circuits, a fairly recent probabilistic model thatwe integrate as the surrogate model in our framework. By enabling users to providevaluable insights, our method harnesses the power of interaction to accelerate theoptimization toward better solutions. We demonstrate the benefits of our approachthrough an empirical evaluation on diverse benchmarks, including the challengingsetting of neural architecture search.

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