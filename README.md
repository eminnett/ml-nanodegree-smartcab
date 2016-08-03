# Project 4: Train a Smartcab to Drive
## Reinforcement Learning
#### Machine Learning Nanodegree (Udacity)
Project submission by Edward Minnett (ed@methodic.io).

August 2nd 2016. (Revision 1)

-------

### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed

### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project.

### Grid Search Results

Udacity has a hard limit on GitHub project submissions that prevents submitting a repository with more than 1000 files. The grid search generated 1024 files as a result, the gridSearch results have not all been committed to the repository. However, the results for alphas 0.01, 0.5, and 0.7 remain in the repository as they contain the optimal result and are required to execute the IPython notebook.

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```

This will run the `agent.py` file and execute your agent code.

### Generate the Report PDF

Run the following in the terminal from the project root:

```
$ ipython nbconvert smartcab_analysis.ipynb --to pdf
$ mv smartcab_analysis.pdf report.pdf
```
