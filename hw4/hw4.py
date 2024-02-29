# %% [markdown]
# # Homework #4
# **Name:** Hanwen Li
# **Email:** lihanwen@seas.upenn.edu  

# %%
import numpy as np
def get_divider(id):
    print('**************************************************** Problem {} ****************************************************'.format(id))

# %% [markdown]
# ## Problem#2

# %%
class Problem1():
  def __init__(self):
    get_divider(1)

    self.mean = 8
    self.dev = 2

  def get_interval(self):
    return np.random.normal(self.mean, self.dev)

  def simulate(self, num):
    interval_times = []
    for i in range(num):
      interval_time = 0
      for i in range(9, 16):
        interval_time += (self.get_interval())
      interval_times.append(interval_time)
    return np.mean(np.array(interval_times) <= 55)

# %%
problem = Problem1()

print("The result of 3000 simulations is {:.4f}".format(problem.simulate(3000)))

# %% [markdown]
# ### Result
# The result of 3000 simulations is 0.4177

# %% [markdown]
# ## Problem#3

# %%
class Problem3():
  def __init__(self):
    get_divider(3)
    self.pro = [0.1, 0.2, 0.3, 0.4]
    self.values = [1/3, 1/2, 2/3, -1]
    self.P = None

  def get_P(self):
    P = np.random.choice(self.values, p=self.pro)
    if P == -1:
      self.P = np.random.uniform()
    else:
      self.P = P

  def simulate(self, num):
    Ps = []
    for i in range(num):
      self.get_P()
      Ps.append(self.P)
    return np.mean(Ps)

# %%
problem = Problem3()

print("The result of 3000 simulations is {:.3f}, the theoretical result is {:.3f}".format(problem.simulate(3000), 8/15))

# %% [markdown]
# ### Result
# The result of 3000 simulations is 0.534, the theoretical result is 0.533

# %% [markdown]
# ## Problem#4

# %%
class Problem4():
  def __init__(self, num):
    get_divider(4)
    self.X = None
    self.Y = None
    self.num = num

  def get_lifetime(self):
    self.X = np.random.exponential(1000, self.num)

  def get_checktime(self):
    self.Y = np.random.geometric(1/1500, self.num)

  def simulate(self):
    self.get_lifetime()
    self.get_checktime()
    return np.mean(self.Y >= self.X)

# %%
problem = Problem4(5000)

print("The result of 5000 simulations is {:.3f}".format(problem.simulate()))

# %% [markdown]
# ### Result
# The result of 5000 simulations is 0.591

# %% [markdown]
# ## Problem#6

# %%
class Problem6():
  def __init__(self):
    get_divider(6)
    self.a = None
    self.b = None
    self.X = None
    self.Y = None

  def get_XY(self):
    self.X = np.random.uniform(self.a, self.b)
    self.Y = np.random.uniform(self.a, self.b)

  def set_ab(self, a, b):
    self.a = a
    self.b = b
  
  def cdf_z(self, z):
    if self.a**2 <= z <= self.a*self.b:
        return (z * np.log(z/self.a**2) - z + self.a**2)/(self.b-self.a)**2
    elif self.a*self.b < z <= self.b**2:
        return (z * np.log(self.b**2/z) + z - 2*self.a*self.b + self.a**2)/(self.b-self.a)**2

  def simulate(self, a, b, c, d, num=4000):
    self.set_ab(a, b)
    theory = self.cdf_z(d) - self.cdf_z(c)
    s = []
    areas = []
    for i in range(num):
      self.get_XY()
      areas.append(self.X*self.Y)
      if c <= self.X*self.Y <= d:
        s.append(1)
      else:
        s.append(0)
    print("The theoretical result is {:.3f} and the simulation result is {:.3f}. And the expectation of area is {:.3f} while the theoretical result is {}".format(theory, np.mean(s), np.mean(areas),((a+b)/2)**2))


# %%
problem = Problem6()

problem.simulate(50, 100, 3000, 8000)
problem.simulate(10, 20, 200, 300)
problem.simulate(300, 1000, 300*600, 300*800)

# %% [markdown]
# ### Result
# The theoretical result is 0.895 and the simulation result is 0.899. And the expectation of area is 5608.736 while the theoretical result is 5625.0
# 
# The theoretical result is 0.477 and the simulation result is 0.470. And the expectation of area is 225.943 while the theoretical result is 225.0
# 
# The theoretical result is 0.103 and the simulation result is 0.095. And the expectation of area is 424878.891 while the theoretical result is 422500.0

# %% [markdown]
# ## Simulation Problem #2

# %%
class SimulationProblem:
    def __init__(self):
        get_divider('Simulation 2')

        self.max_capacity = 12
        self.floors = [i+1 for i in range(10)]

        self.current = None

    def get_on(self):
        self.current += np.random.choice([i for i in range(self.max_capacity-self.current+1)])
        
    def get_off(self):
        self.current -= np.random.choice([i for i in range(self.current+1)])

    def simulate(self, num):
        people = []
        for n in range(num):
            self.current = 0
            for m in (self.floors + self.floors):
                self.get_off()
                self.get_on()
                people.append(self.current)
        
        print("The average number of people on the elevator at any given time is {:.2f}".
              format(np.mean(people)))
        
        print('The probability that the elevator is empty at any given time is {:.2f}'.
              format(np.mean(np.array(people)==0)))
        
        print('The probability that the elevator is filled at any given time is {:.2f}'.
              format(np.mean(np.array(people)==self.max_capacity)))

# %%
problem = SimulationProblem()

problem.simulate(1000)

# %% [markdown]
# ### Result
# The average number of people on the elevator at any given time is 7.88
# 
# The probability that the elevator is empty at any given time is 0.02
# 
# The probability that the elevator is filled at any given time is 0.14


