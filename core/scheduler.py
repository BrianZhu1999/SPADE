"""Q-learning allocates unevaluated blocks among generic operator streams."""
import numpy as np

class Scheduler:
    def __init__(self,seed,mode='q_learning'):
        self.rng=np.random.default_rng(seed);self.mode=mode;self.q={};self.visits={};self.step=0;self.log=[]
    def choose(self,families,stage,progress):
        state=(stage,min(int(progress*5),4));self.state=state
        unseen=[f for f in families if (state,f) not in self.visits]
        if self.mode=='round_robin':chosen=families[self.step%len(families)]
        elif self.mode=='random' or unseen or self.rng.random()<.2:
            options=unseen if unseen and self.mode=='q_learning' else families
            chosen=options[int(self.rng.integers(len(options)))]
        else:
            best=max(self.q.get((state,f),0) for f in families)
            ties=[f for f in families if self.q.get((state,f),0)==best]
            chosen=ties[int(self.rng.integers(len(ties)))]
        self.step+=1;return chosen
    def update(self,family,reward,evaluated,available):
        key=(self.state,family);old=self.q.get(key,0)
        future=max((self.q.get((self.state,f),0) for f in available),default=0)
        new=old+.2*(float(reward)+.8*future-old)
        if self.mode=='q_learning':self.q[key]=new
        self.visits[key]=self.visits.get(key,0)+1
        self.log.append({'state':list(self.state),'family':family,'reward':float(reward),'evaluated':evaluated,'q_before':old,'q_after':new if self.mode=='q_learning' else old})
