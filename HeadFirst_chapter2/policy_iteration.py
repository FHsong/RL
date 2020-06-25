from HeadFirst_chapter2.grid_mdp import GridEnv

class dynamic_programming:
    def __init__(self):
        pass

    # Policy iteration = policy evaluate + policy improve
    def policy_iteration(self, grid_mdp):
        for i in range (100):
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)


    def policy_evaluate(self, grid_mdp):
        for i in range(10000):
            delta = 0.
            for state in grid_mdp.state_space:
                if state in grid_mdp.terminal_states: continue




    def policy_improve(self, grid_mdp):
        pass
