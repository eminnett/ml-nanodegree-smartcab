import random
import operator
import pandas as pd
import time
import datetime
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_states = {}
        self.n_states = {}
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.5
        self.trial_stats_columns = ['total_reward', 'negative_reward', 'trial_length', 'reached_destination']
        self.trial_stats = pd.DataFrame(columns=self.trial_stats_columns)
        self.actions = ['forward', 'right', 'left', None]
        self.possible_states = self.state_permutations()
        self.verbose_debugging = False

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = {}
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.total_reward = 0
        self.negative_reward = 0
        self.trial_length = 0
        self.reached_destination = False

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.update_state(inputs)

        # TODO: Select action according to your policy
        action = self.policy(self.state, self.exploration_probability(deadline))

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.total_reward += reward
        if reward < 0: self.negative_reward += reward
        self.trial_length += 1
        self.reached_destination = reward > 2

        if self.reached_destination or deadline == 0: self.save_trial_stats()

        # TODO: Learn policy based on state, action, reward
        if self.prev_state != None:
            self.verbose_output("\n\nUpdate Q and N:")

            n_val = self.N_increment(self.prev_state, self.prev_action)
            old_q_val = self.Q_get(self.prev_state, self.prev_action)
            new_q_val = ((1 - self.alpha) * old_q_val
                + self.alpha * (self.prev_reward + self.gamma * self.Q_max(self.state)))
            self.Q_set(self.prev_state, self.prev_action, new_q_val)

            self.verbose_output("Previous State: {}".format(self.state_string(self.prev_state)))
            self.verbose_output("Previous Action: {}".format(self.prev_action))
            self.verbose_output("Q(s,a):")
            self.verbose_output(self.state_action_matrix_string(self.Q_get))
            self.verbose_output("N(s,a):")
            self.verbose_output(self.state_action_matrix_string(self.N_get))

        self.prev_state = self.state.copy()
        self.prev_action = action
        self.prev_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def update_state(self, inputs):
        self.state['env'] = inputs
        self.state['desired_direction'] = self.next_waypoint

    def exploration_probability(self, deadline):
        n_max = max(1, self.N_max(self.state))
        eagerness_to_explore = self.epsilon * deadline / n_max
        return min(1, eagerness_to_explore)

    # The Q-Learner Policy
    def policy(self, s, exploration_probability):
        return self.next_waypoint
        # self.verbose_output(("------------------------------------\n"
        #     + "policy(s):\n"
        #     + "Exploration Probability: {}").format(exploration_probability))
        #
        # if random.uniform(0, 1) < exploration_probability:
        #     self.verbose_output("Exploring")
        #     action = random.choice(self.actions)
        # else:
        #     state_string = self.state_string(s)
        #     q_values = self.q_states[state_string]
        #     self.verbose_output("state_string: {}".format(state_string))
        #     self.verbose_output("Q Values for state: {}".format(q_values))
        #     sorted_q_value_tuples = sorted(q_values.items(), key=operator.itemgetter(1), reverse=True)
        #     q_max_tuple = sorted_q_value_tuples[0]
        #     action = q_max_tuple[0]
        # self.verbose_output("Chosen action: {}".format(action))
        # return action

    def Q_get(self, s, a):
        state_string = self.state_string(s)
        if state_string in self.q_states and a in self.q_states[state_string]:
            return self.q_states[state_string][a]
        return 0

    def Q_set(self, s, a, v):
        state_string = self.state_string(s)
        if state_string in self.q_states:
            self.q_states[state_string][a] = v
        else:
            self.q_states[state_string] = {a: v}

    def Q_max(self, s):
        return sorted([self.Q_get(s, a) for a in self.actions], reverse=True)[0]

    def N_get(self, s, a):
        state_string = self.state_string(s)
        if state_string in self.n_states and a in self.n_states[state_string]:
                return self.n_states[state_string][a]
        return 0

    def N_increment(self, s, a):
        state_string = self.state_string(s)
        if state_string in self.n_states:
            if a in self.n_states[state_string]:
                self.n_states[state_string][a] += 1
            else:
                self.n_states[state_string][a] = 1
        else:
            self.n_states[state_string] = {a: 1}
        return self.n_states[state_string][a]

    def N_max(self, s):
        return sorted([self.N_get(s, a) for a in self.actions], reverse=True)[0]

    def state_string(self, s):
        tl = s['env']['light']
        o = s['env']['oncoming']
        r = s['env']['right']
        l = s['env']['left']
        dd = s['desired_direction']
        return "tl:{},o:{},r:{},l:{},dd:{}".format(tl, o, r, l, dd)

    def state_action_matrix_string(self, getter):
        """
        State               | forward  | right    | left     | None     |
        ct:True,dd:forward  | 53       | 8        | 15       | 19       |
        ct:True,dd:right    | 18       | 70       | 11       | 13       |
        ct:True,dd:left     | 7        | 8        | 7        | 9        |
        ct:False,dd:forward | 17       | 12       | 20       | 70       |
        ct:False,dd:right   | 0        | 0        | 0        | 0        |
        ct:False,dd:left    | 9        | 14       | 8        | 4        |
        """
        longest_state_string = 49
        value_length = 8
        output = "{} |".format(self.fixed_length_string("State", longest_state_string))
        for a in self.actions:
            output += " {} |".format(self.fixed_length_string(str(a), value_length))
        output += "\n"
        for s in self.possible_states:
            s_string = self.fixed_length_string(self.state_string(s), longest_state_string)
            output += "{} |".format(s_string)
            for a in self.actions:
                a_string = self.fixed_length_string(str(getter(s, a)), value_length)
                output += " {} |".format(a_string)
            output += "\n"
        return output

    def fixed_length_string(self, string, length):
        if len(string) < length:
            return string + (' ' * (length - len(string)))
        elif len(string) > length:
            return string[:length]
        return string

    def state_permutations(self):
        light = ['green', 'red']
        oncoming = self.actions
        right = self.actions
        left = self.actions
        desired_directions = ['forward', 'right', 'left']
        states = []
        for tl in light:
            for o in oncoming:
                for r in right:
                    for l in left:
                        for dd in desired_directions:
                            states.append({
                                    'env': {'light': tl, 'oncoming': o,'right': r,'left': l},
                                    'desired_direction': dd
                                })
        return states

    def verbose_output(self, string):
        if self.verbose_debugging:
            print string

    def save_trial_stats(self):
        trial_data = [self.total_reward, self.negative_reward, self.trial_length, self.reached_destination]
        trial_df = pd.DataFrame([trial_data], columns=self.trial_stats_columns)
        .trial_stats = self.trial_stats.append(trial_df, ignore_index=True)
        if self.trial_stats.shape[0] == 100:
            print "*****\nReporting Data\n*****"
            self.report_data()

    def report_data(self):
        self.trial_stats.to_csv(self.file_name('trial_stats', 'csv'))
        matrices_text_file = open(self.file_name('Q_and_N', 'txt'), "w")
        matrices_text_file.write("Q(s,a):\n")
        matrices_text_file.write(self.state_action_matrix_string(self.Q_get))
        matrices_text_file.write("\n")
        matrices_text_file.write("N(s,a):\n")
        matrices_text_file.write(self.state_action_matrix_string(self.N_get))
        matrices_text_file.close()

    def file_name(self, base, file_type):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        return "./data/{}_{}_q_agent_a:{}_g:{}_e:{}.{}".format(base, st, self.alpha, self.gamma, self.epsilon, file_type)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
