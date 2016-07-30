import random
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
        self.state = {}
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.q_states = {}
        self.n_states = {}
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.5
        self.all_rewards = []
        self.cumulative_reward = 0
        self.actions = ['forward', 'right', 'left', None]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = {}
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        print "Accumulated reward: {}".format(self.cumulative_reward)
        if (len(self.all_rewards) > 0 and len(self.all_rewards) < 100) or self.cumulative_reward != 0:
            self.all_rewards.append(self.cumulative_reward)
        print "Number of iterations: {}".format(len(self.all_rewards))
        print "All Rewards; {}".format(self.all_rewards)
        self.cumulative_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.update_state(inputs)

        # TODO: Select action according to your policy
        action = self.policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.cumulative_reward += reward

        # TODO: Learn policy based on state, action, reward
        if self.prev_state != None:
            print "\n"
            print "Update Q and N:"

            n_val = self.N_increment(self.prev_state, self.prev_action)
            old_q_val = self.Q_get(self.prev_state, self.prev_action)
            new_q_val = ((1 - self.alpha) * old_q_val
                + self.alpha * (self.prev_reward + self.gamma * self.Q_max(self.state)))
            self.Q_set(self.prev_state, self.prev_action, new_q_val)

            print "Previous State: {}".format(self.state_string(self.prev_state))
            print "Previous Action: {}".format(self.prev_action)
            print "Q(s,a):"
            print self.state_action_matrix_string(self.Q_get)
            print "N(s,a):"
            print self.state_action_matrix_string(self.N_get)

        self.prev_state = self.state.copy()
        self.prev_action = action
        self.prev_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def update_state(self, inputs):
        legal_moves = {
            'forward': inputs['light'] == 'green',
            'right': inputs['light'] == 'green' or inputs['left'] != 'forward',
            'left': inputs['light'] == 'green' and (inputs['oncoming'] == None or inputs['oncoming'] == 'left')
        }
        self.state['can_travel_in_desired_direction'] = legal_moves[self.next_waypoint]
        self.state['desired_direction'] = self.next_waypoint

    # The Q-Learner Policy
    def policy(self, s):
        print "------------------------------------"
        print "policy(s):"
        print "state_string: {}".format(self.state_string(s))
        a_by_q = {}
        a_by_n = {}
        for a in self.actions:
            q = self.Q_get(s, a)
            n = self.N_get(s, a)
            if q in a_by_q:
                a_by_q[q].append(a)
            else:
                a_by_q[q] = [a]
            if n in a_by_n:
                a_by_n[n].append(a)
            else:
                a_by_n[n] = [a]
        print "a_by_n: {}".format(a_by_n)
        print "a_by_q: {}".format(a_by_q)
        if bool(set(range(0, 10)) & set(a_by_n.keys())):
            least_used_actions = a_by_n[sorted(a_by_n)[0]]
            action = random.choice(least_used_actions)
        else:
            as_max_q = a_by_q[sorted(a_by_q, reverse=True)[0]]
            if len(as_max_q) == 1:
                action = as_max_q[0]
            else:
                n_by_a_max_q = {n: a for n, a in a_by_n.iteritems() if bool(set(a) & set(as_max_q))}
                as_min_n = n_by_a_max_q[sorted(n_by_a_max_q)[0]]
                action = random.choice(as_min_n)
        print "Chosen action: {}".format(action)
        return action

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
        a_by_q = {}
        for a in self.actions:
            q = self.Q_get(s, a)
            if q in a_by_q:
                a_by_q[q].append(a)
            else:
                a_by_q[q] = [a]
        return sorted(a_by_q, reverse=True)[0]

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

    def state_string(self, s):
        ct = s['can_travel_in_desired_direction']
        dd = s['desired_direction']
        return "ct:{},dd:{}".format(ct, dd)

    def state_action_matrix_string(self, getter):
        output = "State               | forward  | right    | left     | None     |\n"
        for s in self.state_permutations():
            state_string = self.state_string(s)
            if len(state_string) < 19:
                state_string = state_string + (' ' * (19 - len(state_string)))
            output += state_string + " |"
            for a in self.actions:
                value = str(getter(s, a))
                if len(value) < 8:
                    value = value + (' ' * (8 - len(value)))
                elif len(value) > 8:
                    value = value[:8]
                output += " {} |".format(value)
            output += "\n"
        return output

    def state_permutations(self):
        can_travel_options = [True, False]
        desired_directions = ['forward', 'right', 'left']
        states = []
        for ct in can_travel_options:
            for dd in desired_directions:
                states.append({
                        'can_travel_in_desired_direction': ct,
                        'desired_direction': dd
                    })
        return states

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

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
