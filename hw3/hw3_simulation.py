# %% [markdown]
# # Homework #3
# **Name:** Hanwen Li
# **Email:** lihanwen@seas.upenn.edu  

# %%
import numpy as np
def get_divider(id):
    print('**************************************************** Problem {} ****************************************************'.format(id))

# %% [markdown]
# ## Problem #1
# See the pdf

# %% [markdown]
# ## Problem #2

# %%
class Problem2():
    def __init__(self):
        get_divider(2)
        self.X = None
        self.Y = None
        self.P = None
        self.L = None


    def get_cordinate(self):
        self.X = np.random.uniform()
        self.Y = np.random.uniform()

    def get_distance(self):
        return (self.X**2+self.Y**2)**0.5
    
    def get_probability(self, L):
        if L <= 1:
            self.P =  L*L*np.pi/4
        else:
            self.P = np.pi/4*L*L + (L*L-1)**0.5 - L*L*np.arctan((L*L-1)**0.5)

    def simulation(self, num):
        self.L = np.random.uniform(low=0, high=2)**0.5
        self.get_probability(self.L)
        distance = []
        lay_in = []
        for i in range(num):
            self.get_cordinate()
            distance.append(self.get_distance())
            if self.get_distance() <= self.L:
                lay_in.append(1)
            else:
                lay_in.append(0)

        return lay_in, distance


# %%
problem2 = Problem2()
for i in range(10):
    lay_in, distance = problem2.simulation(3000)
    print("The result of simulation is {:.3f}, while the result of calculation is {:.3f}, and the average distance is {:.3f}".format(np.mean(lay_in), problem2.P,np.mean(distance)))

# %% [markdown]
# ### Result
# The result of simulation is 0.980, while the result of calculation is 0.983, and the average distance is 0.764
# 
# The result of simulation is 0.802, while the result of calculation is 0.803, and the average distance is 0.764
# 
# The result of simulation is 0.614, while the result of calculation is 0.617, and the average distance is 0.772
# 
# The result of simulation is 0.534, while the result of calculation is 0.537, and the average distance is 0.768
# 
# The result of simulation is 0.392, while the result of calculation is 0.397, and the average distance is 0.763
# 
# The result of simulation is 0.855, while the result of calculation is 0.856, and the average distance is 0.772
# 
# The result of simulation is 0.866, while the result of calculation is 0.871, and the average distance is 0.764
# 
# The result of simulation is 0.948, while the result of calculation is 0.947, and the average distance is 0.769
# 
# The result of simulation is 0.964, while the result of calculation is 0.963, and the average distance is 0.759
# 
# The result of simulation is 0.993, while the result of calculation is 0.995, and the average distance is 0.767

# %% [markdown]
# ## Problem #3
# See the pdf

# %% [markdown]
# ## Problem #4
# See the pdf

# %% [markdown]
# ## Problem #5

# %%
class Problem5():
    def __init__(self):
        get_divider(5)
        self.A = None
        self.B = None
        self.C = None

    def get_random_variables(self, l):
        self.A =  np.random.exponential(l)
        self.B = np.random.exponential(l)
        self.C = np.random.exponential(l)

    def get_discriminant(self):
        return self.B**2 - 4*self.A*self.C
    
    def simulate(self, num, l):
        real = []
        for i in range(num):
            self.get_random_variables(l)
            if self.get_discriminant() >= 0:
                real.append(1)
            else:
                real.append(0)
        return real

# %%
problem = Problem5()

for l in [0.5, 1, 2, 5, 10]:
    real = problem.simulate(5000, l)
    print("The result of simulation is {:.3f}".format(np.mean(real)))

# %% [markdown]
# ### Result
# The result of simulation is 0.340
# 
# The result of simulation is 0.327
# 
# The result of simulation is 0.337
# 
# The result of simulation is 0.335
# 
# The result of simulation is 0.327

# %% [markdown]
# ## problem #6

# %%
class Problem6():
    def __init__(self):
        get_divider(6)
        self.P = {'A':None, 'B':None}

    def set_probability(self):
        self.P['A'] = np.random.uniform()
        self.P['B'] = np.random.uniform()

    def equation(self, question='a'):
        if question == 'a':
            return self.P['A']/(1-(1-self.P['A'])*(1-self.P['B']))
        else:
            return (2-self.P['A'])/(self.P['A']+self.P['B']-self.P['A']*self.P['B'])
        
    def roll(self, role='A'):
        if role == 'A':
            return np.random.choice([1, 0], p=[self.P['A'], 1 - self.P['A']])
        else:
            return np.random.choice([1, 0], p=[self.P['B'], 1 - self.P['B']])
        
    def simulate(self, num):
        self.set_probability()
        rolls = []
        a_wins = []
        for i in range(num):
            rolls.append(0)
            while True:
                rolls[i] += 1
                if self.roll('A') == 1:
                    a_wins.append(1)
                    break
                else:
                    rolls[i] += 1
                    if self.roll('B') == 1:
                        a_wins.append(0)
                        break

        return rolls, a_wins
            

# %%
problem = Problem6()

for i in range(5):
    rolls, a_wins = problem.simulate(5000)
    print("The result of simulation is rolls : {:.3f} and A wins rolls, a_wins: {:.3f}".format(np.mean(rolls), np.mean(a_wins)))
    print("The result of calculation is rolls : {:.3f} and A wins rolls, a_wins: {:.3f}".format(problem.equation('b'), problem.equation('a')))
    print('-'*50)

# %% [markdown]
# ### Result
# The result of simulation is rolls : 1.206 and A wins rolls, a_wins: 0.979
# 
# The result of calculation is rolls : 1.215 and A wins rolls, a_wins: 0.976
# **************************************************
# The result of simulation is rolls : 3.061 and A wins rolls, a_wins: 0.083
# 
# The result of calculation is rolls : 3.052 and A wins rolls, a_wins: 0.083
# **************************************************
# The result of simulation is rolls : 1.987 and A wins rolls, a_wins: 0.113
# 
# The result of calculation is rolls : 1.991 and A wins rolls, a_wins: 0.117
# **************************************************
# The result of simulation is rolls : 1.966 and A wins rolls, a_wins: 0.582
# 
# The result of calculation is rolls : 1.947 and A wins rolls, a_wins: 0.582
# **************************************************
# The result of simulation is rolls : 4.196 and A wins rolls, a_wins: 0.774
# 
# The result of calculation is rolls : 4.208 and A wins rolls, a_wins: 0.773
# **************************************************

# %% [markdown]
# ## Simulation Problem #6

# %%
class SimulationProblem:
  def __init__(self):
    get_divider('Simulation')
    self.max_capacity = 250
    self.stock = 250
    self.demand = None
    self.supply = 0
    self.profit = 10
    self.storage_cost = 6

  def set_demand(self):
    self.demand = round(np.random.uniform(low=24.5, high=50.5))

  def set_supply(self):
    self.supply = round(np.random.uniform(low=0.5, high=5.5))

  def get_daily_profit(self):
    profit = 0
    if self.demand > self.stock:
      profit += self.stock*self.profit
      profit -= (self.demand-self.stock)*self.profit
      self.stock = 0
    else:
      profit += self.demand*self.profit
      self.stock -= self.demand

    return profit

  def get_storage_cost(self, arrive=False):
    if arrive:
      self.stock += self.max_capacity
      self.supply = -1

    return -(self.stock - self.max_capacity)*self.storage_cost if self.stock > self.max_capacity else 0

  def simulate(self, num, l):
    profits = []
    for i in range(num):
      self.set_demand()
      profit = self.get_daily_profit()
      if self.supply > 1:
        self.supply -= 1
        profit += self.get_storage_cost()
      elif self.supply == 1:
        profit += self.get_storage_cost(arrive=True)
      else:
        if self.stock < l*self.max_capacity:
          self.set_supply()
          profit += self.get_storage_cost()
      profits.append(profit)

    return profits

# %%
problem = SimulationProblem()

for i in range(21):
  profits = problem.simulate(10000, i/20)
  print("The result of simulation with lambda {} is {:.1f}".format(i/20, np.mean(profits)))

# %% [markdown]
# ### Result
# The max profit is achieved when lambda is in [0.65, 0.7]


