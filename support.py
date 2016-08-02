import glob
import pandas as pd
import numpy as np
from IPython.display import display
import seaborn as sns

def load_df(file_name_pattern):
    """
    Read the first CSV file that matches the given file name pattern and return
    the DataFrame that represnts that CSV file.
    """
    file_names = glob.glob(file_name_pattern)
    if len(file_names) > 0:
        df = pd.read_csv(file_names[0])
        df.rename(columns={'Unnamed: 0': 'Trial'}, inplace=True)
        return df
    return pd.DataFrame()

def display_trial_stats(df, title_prefix, ylim_bottom, ylim_top):
    """
    Display summary statistics and time series plot describing the 4 columns in
    a simulation stats DataFrame: The length (number of steps) in each trial of
    the simulation, the total reward for each trial, the total negative reward
    in each trial, and whether each trial reach the designated destination.
    """
    successes = df[df.reached_destination==True].Trial
    failures = df[df.reached_destination==False].Trial

    print "The destination was reached in {} out of {} trials.".format(successes.shape[0], df.shape[0])
    display(df[['total_reward', 'negative_reward', 'trial_length']].describe().T)

    sns.set(font_scale=1.5, style={"axes.facecolor": "white"})
    sns.plt.figure(figsize=(16, 8))
    ax = sns.tsplot(df.trial_length, color='.75', legend=True, condition='Trial Length')
    ax = sns.tsplot(df.total_reward, color='#106B70', legend=True, condition='Total Reward')
    ax = sns.tsplot(df.negative_reward, color='#D43500', legend=True, condition='Negative Reward')
    ax = sns.rugplot(successes, color='green', height=1, linewidth=10, alpha=0.1)
    ax = sns.rugplot(failures, color='red', height=1, linewidth=10, alpha=0.1)
    sns.plt.legend(labels=['Trial Length', 'Total Reward', 'Negative Reward', 'Reached Destination'], frameon=True)
    ax.set(xlabel='Trial', ylabel='Value')
    ax.set_title(title_prefix + ': Trial Length, Total Reward, and Negative Reward for each Trial')
    sns.plt.ylim(ylim_bottom, ylim_top)
    sns.plt.plot([0, 100], [0, 0], linewidth=1, color='.5')

def display_random_agent_stats():
    """
    Display the trial statistics for the Random Action Agent simulation.
    """
    df = load_df("./data/trial_stats_*_random_agent.csv")
    display_trial_stats(df, 'Random Action Agent', -30, 70)

def display_naive_agent_stats():
    """
    Display the trial statistics for the Naive Agent simulation.
    """
    df = load_df("./data/trial_stats_*_naive_agent.csv")
    display_trial_stats(df, 'Naive Agent', -25, 45)

def display_informed_driver_agent_stats():
    """
    Display the trial statistics for the Informed Driver Agent simulation.
    """
    df = load_df("./data/trial_stats_*_informed_driver_agent.csv")
    display_trial_stats(df, 'Informed Driver Agent', -5, 45)

def display_stats_for_the_q_learning_agent_with_params(value):
    """
    Display the trial statistics for the Q-Learning Agent simulation where
    alpha, gamma, and epsilon are all set to the given value.
    """
    df = load_df("./data/gridsearch/alpha_{}/*_g:{}_e:{}.csv".format(value, value, value))
    display_trial_stats(df, "Q-Learning Agent: a, g, and e set to {}".format(value), -50, 70)

def penalty_score(df):
    """
    Calculate the simulation penalty score for the given DataFrame. The larger
    the score, the less scussful the simulation was.
    """
    last_trials_with_negative_reward = df[df.negative_reward < 0].Trial.tolist()[-2:]
    avg_last_trials_with_negative_reward = np.average(last_trials_with_negative_reward)
    last_trial_failures = df[df.reached_destination == False].Trial.tolist()[-2:]
    avg_last_trial_failures = np.average(last_trial_failures)
    return avg_last_trials_with_negative_reward + avg_last_trial_failures

def find_optmal_parameters():
    """
    Search through the results of the gridsearch for alpha, gamma, and epsilon
    and return the DataFrame and parameters for the simulation with the smallest
    penalty score.
    """
    search_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]
    lowest_penalty_score = 10000
    optimal_parameters = {'alpha':0, 'gamma':0, 'epsilon':0}
    optimal_df = None

    for a in search_values:
        for g in search_values:
            for e in search_values:
                file_name_pattern = "./data/gridsearch/alpha_{}/*_g:{}_e:{}.csv".format(a, g, e)
                df = load_df(file_name_pattern)
                if not df.empty:
                    penalty = penalty_score(df)
                    if penalty < lowest_penalty_score:
                        lowest_penalty_score = penalty
                        optimal_df = df
                        optimal_parameters['alpha'] = a
                        optimal_parameters['gamma'] = g
                        optimal_parameters['epsilon'] = e

    return optimal_df, optimal_parameters

def remove_empty_rows(df, columns):
    return df[(df[columns].T != 0).any()]

def optimal_q_and_n_less_empty_rows():
    """
    Display the Q(s,a) and N(s,a) matrices built up during the optimal
    Q-Learning simulation.
    """
    truncator = lambda x: round(x, 3)
    numeric_columns = ['forward', 'left', 'right', 'None']
    Q_sparse = remove_empty_rows(load_df("./data/Q_optimal_*.csv"), numeric_columns)
    Q_sparse[numeric_columns] = Q_sparse[numeric_columns].applymap(truncator)

    N_sparse = remove_empty_rows(load_df("./data/N_optimal_*.csv"), numeric_columns)

    print "State encoding:"
    print "    tl: Traffic light"
    print "    o:  Oncoming traffic"
    print "    r:  Traffic coming from the right"
    print "    l:  Traffic coming from the left"
    print "    dd: Desired direction"
    print "NB: Please note that states that were not experienced by the agent are not displayed.\n"
    print "Q(s,a):"
    display(Q_sparse)
    print "N(s,a):"
    display(N_sparse)
