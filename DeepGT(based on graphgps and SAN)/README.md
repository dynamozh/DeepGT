## Installation
To setup the Python environment, please install conda first. 
All the required environments are in [requirements1.txt](./requirements1.txt) and [requirement2.yml](./requirement2.yml).

## How to Run

To run the experiments, please refer to the commands below:
```bash
# Run 10 repeats with 10 different random seeds:
python main.py --cfg configs/GPS/imdb-binary-GPS+RWSE.yaml  --repeat 10 seed 3407
# Run a particular random seed:
python main.py --cfg configs/GPS/imdb-binary-GPS+RWSE.yaml  --repeat 1 seed 3407
```
