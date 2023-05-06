# Processing of Full Factorial Experiment (FFE)
[Factorial Experiment](https://en.wikipedia.org/wiki/Factorial_experiment)

This py code contains interface for reading input data (factors and experiment results for FFE) and performs analysis of this data: it builds FFE-table, performs calculations of mean and dispersion for experiments, finds coefficients for polinomial linear model and performs significance and adequacy tests for found coefficients, optimizes found polinomial function. 

## Interface
Program can read input data from csv-files or from console input. Program assumes that user knows what he is doing, so there is no much of input validation, still program allows to use input data that doesn't fully satisfies the definition of FFE.

At first step of program user is asked if he wants to read input from files.

### Reading input from files
Example of samples can be found in `samples` folder.
1. Pick csv-file with data for factors part of FFE-table. Each row in file should represent factor:
```
    [x1_exp1],[x1_exp2],...,[x1_expN]
    [x2_exp1],[x2_exp2],...,[x2_expN]
    ...
    [xK_exp1],[xK_exp2],...,[xK_expN]
```
- K - number of factors
- N - 2^K, number of experiments

2. Pick csv-file with data for experiments part of FFE-table. Each row in file should represent parallel experiment:
```
    [y1_exp1],[y1_exp2],...,[y1_expN]
    [y2_exp1],[y2_exp2],...,[y2_expN]
    ...
    [yM_exp1],[yM_exp2],...,[yM_expN]
```
- M - number of parallel experiments
- N - 2^K, number of experiments

### Reading input from console
1. User is asked to fill data about ffe-table sizes:
- K - number of factors
- M - number of parallel experiments

2. User is asked if he wants to fill factors table manually. If yes, he will be asked to fill data in format similar to input from files. If no, he will need to specify main level and changing interval for each factor:

```
    [x1_main_lvl],[x1_chng_int]
    [x2_main_lvl],[x2_chng_int]
    ...
    [xK_main_lvl],[xK_chng_int]

```
3. User is asked to fill experiment results table in format similar to file input.

### Transforming data
If some of data doesn't correspond to FFE, user will receive warning in console, that some of data is wrong, and will be transformed to keep data consistent for running program:

- If factors table contains None values for some experiment, cells with data for this experiment will be deleted.
- If factors or experiment results table has more rows than other, last rows in larger table will be deleted.

User will receive warnings if data isn't consistent:
- some values in factors table can't be encoded to 1 or -1;
- some values in experiments table are None;
- size of table N (experiment number) is less than 2^K.

In all these cases program will try to calculate some data for FFE, but don't expect them to be correct.

### Running tests
If there is more than 1 parallel experiment, user will be asked if he wants to run significance and/or adequacy tests for found coefficients.

## FFE processing

1. **Calculating coded factor values and experiment mean.** Factors in natural values will be encoded to -1/1 values. For experiment results, if there is more than 1 parallel experiment, mean and dispersion will be calculated.
2. **Calculating coefficients for polinomial model.**
3. **Running significance test.** For coefficients found in previous step run t-test to determine if coefficients are significant for p=0.95.
4. **Running adequacy test.** For found polinomial coefficients run f-test to determine if model is adequate for p=0.95.
5. **Calculate polinomial coefficients in natural values.**
6. **Running optimization for found linear model.** Finding x values for minumum linear model value.
 