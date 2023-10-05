# %%
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from qibo import gates
from dataclasses import dataclass
from typing import Union
from itertools import combinations, permutations
# rng = np.random.default_rng(seed=123)

@dataclass
class gate_type:
    _gate: gates.Gate
    _indices: Union[int,tuple]

@dataclass
class constants:
    _comb_single = np.array(list(combinations([0,1,2],1))).flatten()
    _comb_tuple = np.array(list(permutations([0,1,2],2)))


class circ_decomposition(gym.Env):
    def __init__(self):
        # self.grid =  [gate_type(gates.H('_'),2),gate_type(gates.CNOT('_','-'),(1,2)),gate_type(gates.TDG('_'),2),
        #         gate_type(gates.CNOT('_','-'),(0,2)),gate_type(gates.T('_'),2),gate_type(gates.CNOT('_','-'),(1,2)),
        #         gate_type(gates.TDG('_'),2),gate_type(gates.CNOT('_','-'),(0,2)),gate_type(gates.T('_'),2),
        #         gate_type(gates.TDG('_'),1),gate_type(gates.H('_'),2),gate_type(gates.CNOT('_','-'),(0,1)),
        #         gate_type(gates.TDG('_'),1),gate_type(gates.CNOT('_','-'),(0,1)),gate_type(gates.S('_'),1),
        #         gate_type(gates.T('_'),0)]
        self.initial_grid =  [gate_type(gates.T('_'),0),gate_type(gates.CNOT('_','-'),(2,1)),gate_type(gates.S('_'),0),
                gate_type(gates.CNOT('_','-'),(1,0)),gate_type(gates.TDG('_'),0),gate_type(gates.CNOT('_','-'),(2,1)),
                gate_type(gates.S('_'),0),gate_type(gates.CNOT('_','-'),(1,0)),gate_type(gates.TDG('_'),0),
                gate_type(gates.S('_'),2),gate_type(gates.T('_'),0),gate_type(gates.CNOT('_','-'),(1,2)),
                gate_type(gates.S('_'),2),gate_type(gates.CNOT('_','-'),(1,2)),gate_type(gates.T('_'),2),
                gate_type(gates.TDG('_'),1)]
        self.grid = self.initial_grid
        self.start = (0)
        self.goal = (self.grid.__len__()-1)
        self.position = np.random.randint(0,self.grid.__len__())
        self.action_space = gym.spaces.Discrete(19) # 3xH, 3xS, 3xT, 3xTDGG, 6xCNOT
        self.observation_space = gym.spaces.Discrete(16)
        self.toffoli_hs_goal = np.trace(np.conj(gates.TOFFOLI(0,1,2).matrix()).T@gates.TOFFOLI(0,1,2).matrix())
        self.count = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.max_length = 2000
        self.epsilon = 0.1

    def step(self, action):
        self.count += 1
        info = {}
        if self.count>self.max_length:
            self.truncated = True
            reward = -1
            return self.position, reward, self.terminated, self.truncated, info
        
        pos = self.position
        match action:
            case 0 | 1 | 2:
                if self.grid[pos]._gate.name != 'h':
                    self.grid[pos]._gate = gates.H('_')
            case 3 | 4 | 5:
                if self.grid[pos]._gate.name != 't':
                    self.grid[pos]._gate = gates.T('_')
            case 6 | 7 | 8:
                if self.grid[pos]._gate.name != 's':
                    self.grid[pos]._gate = gates.T('_')
            case 9 | 10 | 11:
                if self.grid[pos]._gate.name != 'tdg':
                    self.grid[pos]._gate = gates.TDG('_')
            case 12 | 13 | 14 | 15 | 16 | 17:
                if self.grid[pos]._gate.name != 'cx':
                    self.grid[pos]._gate = gates.CNOT('_','-')
            case 18:
                pass
            case _:
                print('step function is faulty')

        if action < 12:
            a = action%3
            self.grid[pos]._indices = a
        elif action < 18:
            a = action%6
            self.grid[pos]._indices = tuple(constants._comb_tuple[a])
        
        reward = 0
        Op = self.circ_matrix()
        # print(Op)
        # print('\n')
        dist = np.abs(np.trace(np.conj(Op).T@gates.TOFFOLI(0,1,2).matrix())/8)
        # print(dist)
        # dist = np.abs(hs_gate - self.toffoli_hs_goal)
        if dist > 0.99:
            # print(Op)
            # print('\n')
            self.terminated = True
            reward = self.max_length - self.count + 1
        else:
            reward = -(1-dist)/self.max_length
        
        self.position = np.random.randint(0,self.grid.__len__())

        # Return the next state, reward, and whether the episode is done
        return self.position, reward, self.terminated, self.truncated, info
    
    def circ_matrix(self,ind=0):
        if ind > self.goal:
            return np.eye(2**3)
        else:
            # print(self.mat_rep(self.grid[ind]))
            return self.mat_rep(self.grid[ind])@self.circ_matrix(ind+1)

    def mat_rep(self,gate_in):
        ket_0 = np.array([[1,0]])
        ket_1 = np.array([[0,1]])
        name = gate_in._gate.name
        if name != 'cx':
            return self.tensor_prod(gate_in)
        else:
            ctr,tgt = gate_in._indices
            state_0 = []
            state_1 = []
            for i in range(0,3):
                if i == ctr:
                    state_0.append(ket_0.T@ket_0)
                    state_1.append(ket_1.T@ket_1)
                elif i == tgt:
                    state_0.append(np.eye(2))
                    state_1.append(gates.X('_').matrix())
                else:
                    state_0.append(np.eye(2))
                    state_1.append(np.eye(2))
            return np.kron(state_0[0], np.kron(state_0[1], state_0[2])) + np.kron(state_1[0], np.kron(state_1[1], state_1[2]))
    
    def tensor_prod(self, gate_in, level=0):
        if level > 2:
            return 1
        else:
            if gate_in._indices == level:
                matrix = gate_in._gate.matrix()
            else:
                matrix = np.eye(2)
        return np.kron(matrix, self.tensor_prod(gate_in,level+1))

    def to_s(self, row, col):
        return row * self.size + col

    def reset(self,seed=None):
        self.position = np.random.randint(0,self.grid.__len__())
        self.grid = self.initial_grid
        self.truncated = False
        self.terminated =False
        self.count = 0
        info = {}
        return self.position,info

    def render(self):
        pass

env = circ_decomposition()
check_env(env)


policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]))
model = PPO('MlpPolicy', env, learning_rate=0.00025, clip_range=0.2, batch_size=128, policy_kwargs=policy_kwargs, verbose=1)

# model = PPO('MlpPolicy', env, verbose=1)

print(model.policy)

model.learn(total_timesteps=100000)
# model.save("frozenlake")

# del model # remove to demonstrate saving and loading

# model = PPO.load("frozenlake")

obs = env.reset()
done = False
# while not done:
#     action, _states = model.predict(obs)
#     print(action)
#     obs, rewards, done, info = env.step(action)
#     print('obs',obs)
#     env.render()



















