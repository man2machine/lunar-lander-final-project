import pickle
from lunar_lander import LunarLander
# from ilqr import sigmoid
import numpy as np
import matplotlib.pyplot as plt
class ILQRPlaybackPolicy:
    def __init__(self, x_trj, u_trj, i=0):
        self.x_trj = x_trj
        self.u_trj = u_trj
        self.i = i
    
    def get_state(self):
        data = {
            'x': self.x_trj,
            'u': self.u_trj,
            'i': self.i
        }
        return data
    
    @classmethod
    def load_state(cls, data):
        return cls(data['x'], data['u'], data['i'])

    def predict(self, s):
        if self.i >= len(self.u_trj):
            raise StopIteration
        u = self.u_trj[self.i]
        self.i += 1
        return u
    
class ILQRModelPredictivePolicy:
    def __init__(self, env):
        self.env = env
        self._sub_policies = [None]
    
    def get_state(self):
        datas = []
        for p in self._sub_policies:
            if p:
                datas.append(p.get_state())
            else:
                datas.append(None)

    def load_state(self, datas):
        self._sub_policies = []
        for d in datas:
            if d:
                self._sub_policies.append(ILQRPlaybackPolicy.load_state(d))
            else:
                self._sub_policies.append(None)


    def predict(self, s):
        policy = self._sub_policies[-1]
        if policy is None:
            x_trj, u_trj, *_ = run_ilqr(self.env, 500)
            policy = ILQRPlaybackPolicy(x_trj, u_trj)
            self._sub_policies[-1] = policy
        try:
            a = policy.predict(s)
        except StopIteration:
            self._sub_policies.append(None)
            return self.predict(s)
        return a

with open("save.pkl", 'rb') as f:
    s = pickle.load(f)

env = LunarLander()
env.seed(0)
env.reset()
policy = ILQRPlaybackPolicy.load_state(s)   
def sigmoid(x, m=np):
    return 1 / (1 + m.exp(-x))
plt.plot(sigmoid(policy.u_trj[:, 0]), color='r', label='0')
plt.plot(sigmoid(policy.u_trj[:, 1]), color='g', label='1')
plt.plot(sigmoid(policy.u_trj[:, 2]), color='b', label='2')
plt.legend()
print(policy.u_trj)
plt.show()
obs = env.get_state()
for i in range(1000):
    action = policy.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()