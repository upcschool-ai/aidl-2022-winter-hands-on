# Session 1
Minimal PyTorch project with a custom dataset, a custom model and a train function. The task is to map random normal noise to 0.
## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session1 python=3.8
```
Then, activate the environment
```
conda activate aidl-session1
```
and install the dependencies
```
pip install -r requirements.txt
```
## Running the project

To run the project, run
```
python session-1/main.py
```
To run the project with different arguments, run
```
python session-1/main.py --n_samples N --n_features N --n_hidden N --n_outputs N --epochs N --batch_size N --lr x
```

### Debugging with VSCode
#### Interpreter
If you are using VSCode, install the Python extension. Then, set up the interpreter to the new virtualenv you just created (`aidl-session1`) by following [these](https://code.visualstudio.com/docs/python/environments#:~:text=To%20do%20so%2C%20open%20the,Settings%2C%20with%20the%20appropriate%20interpreter) instructions. You can find out the path of your interpreter by running `which python` (Unix) or `where python` (Windows) after activating it.
#### Run configuration
Finally, this run configuration will work for you:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Session 1",
            "type": "python",
            "request": "launch",
            "program": "session-1/main.py",
            "console": "integratedTerminal",
            "args": ["--n_samples", "100000", "--n_features", "10", "--n_hidden", "20", "--n_outputs", "5", "--epochs", "10", "--batch_size", "100", "--lr", "0.01"]
        }
    ]
}
```
You should place it in `.vscode/launch.json`. For more information about debugging configurations, check [these](https://code.visualstudio.com/docs/python/debugging) instructions.

### Debugging with PyCharm
#### Interpreter
If you are using PyCharm, set up the interpreter to the new virtualenv you just created (`aidl-session1`) by following [these](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html) instructions. You can find out the path of your interpreter by running `which python` (Unix) or `where python` (Windows) after activating it.

#### Run configuration
Check [here](https://www.jetbrains.com/help/pycharm/creating-and-editing-run-debug-configurations.html).
