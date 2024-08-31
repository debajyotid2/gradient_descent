# Multivariate Linear Regression using Stochastic Gradient Descent

This repository is a demonstration of linear regression solved using low level matrix and linear algebra operations in C and NumPy in Python. 

- [Quickstart in C](#quickstart-in-c)
  * [How to build](#how-to-build)
  * [How to run](#how-to-run)
  * [Compatibility](#compatibility)
- [Quickstart in Python](#quickstart-in-python)
  * [How to build](#how-to-build-1)
  * [How to run](#how-to-run-1)
- [Example usage](#example-usage)
  * [C](#c)
  * [Python](#python)
- [Theory](#theory)
  * [Assumptions](#assumptions)
- [Gradient Descent](#gradient-descent)
  * [Algorithm](#algorithm)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
  * [Algorithm](#algorithm-1)

## Quickstart in C

### How to build

The C source code uses BLAS (Basic Linear Algebra Subsystems) libraries for matrix and vector operations. The implementation specifically used is [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). I have included a helper script called `setup_dependencies.sh` inside `c/scripts` to download and build the specific versions of these dependencies locally.

To install all dependencies, run
```
cd c/scripts
source setup_dependencies.sh <NUMBER_OF_THREADS>
```

Building these requires a C/C++ compiler (`gcc/clang`) and CMake installed. OpenBLAS takes minutes to build, and can be sped up by specifying more than one `<NUMBER_OF_THREADS>` on your machine when sourcing the setup script.

Once dependencies are installed, source the `build.sh` script to build the source and test files.
```
cd c/scripts
source build.sh <NUMBER_OF_THREADS>
```
[Catch2](https://github.com/catchorg/Catch2) is used for testing. CMake downloads and builds Catch2 while also building tests. Testing is enabled by default.

Building Catch2 is also time-consuming. Specifying more than 1 `<NUMBER_OF_THREADS>` will make compilation faster.

### How to run

To run a multivariate linear regression problem with the default settings and stochastic gradient descent solver, please run

```
cd c/build
./run
```

Hyperparameters can be customized from the command line. For more details, please run `./run --help`.

To run unit tests, please run

```
cd c/build
ctest
```

### Compatibility

Code has currently only been tested on Fedora Linux. The dependencies may or may not work as-is with other platforms. Only a Linux-based distribution is supported for now, so Windows users may look into [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

## Quickstart in Python

### How to build

The only dependencies are `numpy` and `matplotlib`. Python 3.10+ is supported for now.
To create a virtual environment, please run
```
python -m venv venv
```
If on Windows, run
```
venv\Scripts\activate.bat   # For cmd.exe
venv\Scripts\Activate.ps1   # For PowerShell
```
If on a Mac or Unix-based OS, run
```
source venv/bin/activate
```
These commands will activate the virtual environment on your machine. Finally, run
```
pip install -r requirements.txt
```
to install all dependencies.

### How to run

To run a multivariate linear regression problem using the default settings, please run
```
cd python
python main.py
```
Hyperparameters can be customized from the command line. They can be viewed using
```
python main.py --help
```
PyTest is used for unit testing. Tests can be run using
```
cd python
pytest
```

## Example usage

### C

The hyperparameters that can be customized are visible in the following output of `./run --help`

```
Usage: run [OPTION...]
A demonstration of linear regression using gradient descent and stochastic
gradient descent.

  -b, --bias[=BIAS]          Bias term
  -B, --batch_size[=BATCH_SIZE]   Batch size
  -f, --test_frac[=TEST_FRAC]   Fraction of data for test set
  -i, --n_iter[=N_ITER]      Number of iterations
  -I, --noise intensity[=NOISE_INTENSITY]
                             Intensity of Gaussian noise to be added
  -M, --n_samples=N_SAMPLES  Number of samples
  -n, --learning_rate[=LEARNING_RATE]
                             Learning rate for the gradient descent
  -N, --n_features=N_FEATURES   Number of features
  -S, --seed[=SEED]          Random number seed
  -t, --tol[=TOL]            Tolerance for convergence
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

The output with default hyperparameters of `./run` looks like this:

```
Arguments:
n_iter = 10000, tol = 0.001000,
n_features = 20, n_samples = 100000
bias = -300.700000, noise_intensity = 2.000000
learning_rate = 0.001000, batch_size = 32
test_frac = 0.200000, seed = 42

It. 100, loss = 225.8889
It. 200, loss = 202.3389
It. 300, loss = 181.5736
It. 400, loss = 163.2489
It. 500, loss = 147.0635
...
It. 9800, loss = 3.3934
It. 9900, loss = 3.3534
It. 10000, loss = 3.3150
Gradient descent took 28.313645 seconds.
MSE: 3.2978
MAE: 1.4716
R-squared: 0.9868
It. 100, loss = 201.0783
It. 200, loss = 180.0591
It. 300, loss = 161.5581
It. 400, loss = 145.2608
It. 500, loss = 130.8926
...
It. 9800, loss = 2.7880
It. 9900, loss = 2.7330
It. 10000, loss = 2.6799
Stochastic gradient descent took 6.151944 seconds.
MSE: 5.1999
MAE: 1.8582
R-squared: 0.9792
```

### Python

The same hyperparameters can be customized in the python code as well, as visible in the following output of `python main.py --help`

```
usage: Linear regression using SGD [-h] [--n_samples N_SAMPLES]
                                   [--n_features N_FEATURES] [--noise NOISE]
                                   [--bias BIAS] [--learning_rate LEARNING_RATE]
                                   [--batch_size BATCH_SIZE] [--test_frac TEST_FRAC]
                                   [--seed SEED] [--max_iter MAX_ITER] [--tol TOL]

A demonstration of linear regression using gradient descent in NumPy.

options:
  -h, --help            show this help message and exit
  --n_samples N_SAMPLES
                        Number of samples
  --n_features N_FEATURES
                        Number of features
  --noise NOISE         Intensity of Gaussian noise to be added
  --bias BIAS           Bias term
  --learning_rate LEARNING_RATE
                        Learning rate for gradient descent
  --batch_size BATCH_SIZE
                        Batch size
  --test_frac TEST_FRAC
                        Fraction of data for test set
  --seed SEED           Random number seed
  --max_iter MAX_ITER   Maximum number of iterations
  --tol TOL             Tolerance for convergence
```
The output with default hyperparameters of `python main.py` (same as that of the C code) looks like this:

```
INFO:root:100: loss = 1179.3173
INFO:root:200: loss = 507.3512
INFO:root:300: loss = 275.4007
INFO:root:400: loss = 170.9328
...
INFO:root:9900: loss = 4.1598
INFO:root:10000: loss = 4.1577
INFO:root:Gradient descent did not converge.
INFO:root:Gradient descent took 15.427083 seconds.
INFO:root:MSE = 4.1435
INFO:root:MAE = 1.6245
INFO:root:R2 = 1.0000
INFO:root:100: loss = 1318.9693
INFO:root:200: loss = 582.8164
INFO:root:300: loss = 379.9656
INFO:root:400: loss = 156.3315
...
INFO:root:9900: loss = 4.5390
INFO:root:10000: loss = 3.5210
INFO:root:Stochastic gradient descent did not converge.
INFO:root:Minibatch stochastic gradient descent took 0.221708 seconds.
INFO:root:MSE = 4.2095
INFO:root:MAE = 1.6373
INFO:root:R2 = 1.0000
```
Losses of both gradient descent (GD) and stochastic gradient descent (SGD) converge to similar values.
<img src="https://github.com/user-attachments/assets/e25267b5-c896-4fe1-ab0a-ee1de321ba74" width=50% height=50%>

## Theory

The general form of the multivariate linear regression model is:

$$
\vec{y} = \beta_0 \vec{1_m} + \beta_1 \vec{x_1} + \beta_2 \vec{x_2} + \dots + \beta_n \vec{x_n} + \epsilon \vec{1_m}
$$

where:
- $\vec{y}$ is the dependent variable (the outcome we want to predict).
- $\vec{x_1}$, $\vec{x_2}$, $\dots$, $\vec{x_n}$ are the independent variables, each of dimension $m$.
- $\beta_0$ is the y-intercept (the value of $y$ when all predictors are 0).
- $\beta_1$, $\beta_2$, $\dots$, $\beta_n$ are the coefficients (slopes) representing the change in $y$ for a unit change in each predictor.
- $\epsilon$ is the error term, representing the deviation of the observed values from the predicted values.
- $\vec{1_m}$ is the $m$-dimensional column vector of ones.

### Assumptions

1. The relationship between the independent and dependent variables is linear.
2. Observations are independent of each other.
3. The residuals (errors) of the model are normally distributed.
4. Independent variables are not too highly correlated with each other.

The coefficients $\beta_1$, $\beta_2$, $\dots$, $\beta_n$  and bias $\beta_0$ are determined by minimizing the cost function $J(\vec{\theta})$ (mean squared error between predicted $\vec{y}$ and actual $\vec{y}$) using gradient descent and stochastic gradient descent.


## Gradient Descent

For a cost function $J(\theta)$, where $\theta$, represents the model parameters, the update rule for Gradient Descent is:

$$
\vec{\theta} = \vec{\theta} - \alpha \cdot \vec{\nabla} J(\vec{\theta})
$$

where:
- $\vec{\theta}$ are the parameters of the model.
- $\alpha$ is the learning rate, a hyperparameter that determines the step size at each iteration.
- $\vec{\nabla} J(\vec{\theta})$ is the gradient of the cost function with respect to $\theta$.

### Algorithm

1. Initialize the parameters $\vec{\theta}$ (often with random values).
2. Compute the gradient $\vec{\nabla} J(\vec{\theta})$.
3. Update the parameters by moving in the direction of the negative gradient.
4. Repeat the process until the change in the cost function is below a predefined threshold (convergence) or a maximum number of iterations is reached.

## Stochastic Gradient Descent

The update rule for Stochastic Gradient Descent is:

$$
\vec{\theta} = \vec{\theta} - \alpha \cdot \vec{\nabla} J(\vec{\theta}; x^{(i)}, y^{(i)})
$$

where:
- $x^{i}$, $y^{i}$ represents a single training example.
- $\nabla J(\vec{\theta}; x^{i}, y^{i})$ is the gradient of the cost function with respect to $\vec{\theta}$ for that single example.

### Algorithm

1. Gather a batch of random examples from training set $(x^{i}, y^{i})$.
3. Compute the gradient for the random batch.
4. Update the parameters $\vec{\theta}$ using the computed gradient.
5. Repeat the process for a set number of epochs or until convergence.

Minibatch stochastic gradient descent is computationally less expensive (significantly) than gradient descent because of the reduced cost of calculating gradient in every epoch.

