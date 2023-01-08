import numpy as np
import threading
import math
from simulatedannealing import *
from hillclimbing import *

class ParallelSearch:
  def __init__(self, initial_config_ls, eval_func, num_in_parallel, simulated_annealing=False, compare_func=maximize_comparison):
    self.initial_config_ls   = initial_config_ls
    self.eval_func           = eval_func
    self.compare_func        = compare_func
    self.num_in_parallel     = num_in_parallel
    self.simulated_annealing = simulated_annealing
    self.hc_neighbour_func   = None
    self.sa_neighbour_func   = None
    self.max_iterations      = 0
    self.iterations_per      = 0
    self.starting_temp       = 0
    self.alpha               = 0
    self.final_temp          = 0
    self.instances           = [None]*num_in_parallel
    self.threads             = []
    self.max_value           = 0
    self.max_config          = None

  def init_hill_climbing(self, neighbour_func, max_iterations=100):
    if self.simulated_annealing: 
      print('Attempting to initialize Hill Climbing for a Simulated Annealing Instance')
      return
    
    self.hc_neighbour_func = neighbour_func
    self.max_iterations    = max_iterations

  def init_simulated_annealing(self, neighbour_func, iterations_per, starting_temp, final_temp, alpha, decrement_func=None):
    if not self.simulated_annealing: 
      print('Attempting to initialize Simulated Annealing for a Hill Climbing Instance')
      return
    
    self.sa_neighbour_func = neighbour_func
    self.iterations_per    = iterations_per
    self.starting_temp     = starting_temp
    self.final_temp        = final_temp
    self.alpha             = alpha
    self.decrement_func    = decrement_func
  
  def run(self):

    if self.simulated_annealing:

      for i in range(self.num_in_parallel):
        if self.decrement_func == None:
          self.instances[i] = SimulatedAnnealing(self.initial_config_ls[i], 
                                                self.eval_func, self.sa_neighbour_func, 
                                                self.starting_temp, self.final_temp, 
                                                self.alpha, self.iterations_per, compare_func=self.compare_func)
        else:
          self.instances[i] = SimulatedAnnealing(self.initial_config_ls[i], 
                                                self.eval_func, self.sa_neighbour_func, 
                                                self.starting_temp, self.final_temp, 
                                                self.alpha, self.iterations_per, 
                                                 decrement_func=self.decrement_func, 
                                                 compare_func=self.compare_func)

      for i in range(self.num_in_parallel):
        self.threads.append(threading.Thread(target=self.instances[i].run()))
      
      for i in range(self.num_in_parallel):
        self.threads[i].join()

        result, config = self.instances[i].get_final_result()

        if self.compare_func(result, self.max_value):
          self.max_value = result
          self.max_config = config

    else:

      for i in range(self.num_in_parallel):
        self.instances[i] = HillClimbing(self.initial_config_ls[i], self.eval_func,
                                         self.hc_neighbour_func, self.max_iterations,
                                         self.compare_func)

      for i in range(self.num_in_parallel):
        self.threads.append(threading.Thread(target=self.instances[i].run()))
      
      for i in range(self.num_in_parallel):
        self.threads[i].join()

        result, config = self.instances[i].get_final_result()

        if self.compare_func(result, self.max_value):
          self.max_value = result
          self.max_config = config