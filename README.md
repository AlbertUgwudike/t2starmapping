# Python Code

## Setup

-   Ensure t2starmapping/python/ is the working directory
-   Install dependencies in t2starmapping/python/requirements.txt

## Generating training data for MLP

-   $ python3 -m Simulator mlp
-   Modify t2starmapping/python/Simulator/training_data.py to adjust n_samples

## Training the MLP

-   You may use the step above to generate training data, or...
-   Use the 65536 samples already generated in t2starmapping/python/data/
-   $ python3 -m MLP train
-   You will observe the charactersitics of the MLP every 25 epochs

## Demo the MLP and simulator

-   $ python3 -m Simulator demo
-   $ python3 -m MLP demo
