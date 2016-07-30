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

        if len(self.all_rewards) % 10 == 0:
            print self.q_states
            print self.n_states

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
        # Q(s,a) = R(s) + gama * sum_over_s_prime(T(a,a,s_prime)) * max_given_a_prime(Q(s_prime, a_prime))
        # Q_hat(s,a) <-(alpha) r + gamma * max_given_a_prime(Q_hat(s,a))
        # V <-(alpha) X => V <- (1-alpha) * V + alpha * X
        # Alpha => learning rate (should decay over time)
        # tranistion = (state, action, reward, state_prime)
        if self.prev_state != None:
            n_val = self.N_increment(self.prev_state, self.prev_action)
            old_q_val = self.Q_get(self.prev_state, self.prev_action)
            # Fix this to actually use q max for any action
            new_q_val = (1 - self.alpha) * old_q_val + self.alpha * (self.prev_reward + self.gamma * self.Q_get(self.state, action))
            self.Q_set(self.prev_state, self.prev_action, new_q_val)

        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def update_state(self, inputs):
        self.state['can_travel_in_direction'] = {
            'forward': inputs['light'] == 'green',
            'right': inputs['light'] == 'green' or inputs['left'] != 'forward',
            'left': inputs['light'] == 'green' and (inputs['oncoming'] == None or inputs['oncoming'] == 'left')
        }
        self.state['desired_direction'] = self.next_waypoint

    # The Q-Learner Policy
    def policy(self, s):
        actions = [None, 'forward', 'right', 'left']
        a_by_q = {}
        a_by_n = {}
        for a in actions:
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
        if 0 in a_by_n:
            return random.choice(a_by_n[0])
        as_max_q = a_by_q[sorted(a_by_q, reverse=True)[0]]
        if len(as_max_q) == 1:
            return as_max_q[0]
        n_by_a_max_q = {n: a for n, a in a_by_n.iteritems() if bool(set(a) & set(as_max_q))}
        as_min_n = n_by_a_max_q[sorted(n_by_a_max_q)[0]]
        return random.choice(as_min_n)

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
                return self.n_states[state_string][a]
            else:
                self.n_states[state_string][a] = 1
        else:
            self.n_states[state_string] = {a: 1}
        return 1

    def state_string(self, s):
        f = s['can_travel_in_direction']['forward']
        r = s['can_travel_in_direction']['right']
        l = s['can_travel_in_direction']['left']
        dd = s['desired_direction']
        return "f:{},r:{},l:{},dd:{}".format(f, r, l, dd)

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
