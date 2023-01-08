import random
import math

def standard_reduction(temperature, alpha):
  temperature *= alpha
  return temperature

def maximize_comparison(a, b):
  return a > b

class SimulatedAnnealing:
  def __init__(self, initial_config, eval_func, neighbour_func, temp, final_temp, alpha, iterations_per=20, decrement_func=standard_reduction, compare_func=maximize_comparison):
    self.cur_config     = initial_config
    self.max_config     = initial_config
    self.eval_func      = eval_func
    self.neighbour_func = neighbour_func
    self.decrement_func = decrement_func
    self.compare_func   = compare_func
    self.temp           = temp
    self.final_temp     = final_temp
    self.alpha          = alpha
    self.iterations_per = iterations_per
    self.cur_value      = self.initial_value
    self.cur_max        = self.initial_value

  def complete(self):
    return self.temp <= self.final_temp

  def accept_bad_move(self, val_1, val_2):
    p = math.e^(-(val_2 - val_1)/self.temp)
    return random.random() <= p

  def get_final_result(self):
    return self.cur_max, self.max_config

  def run(self):

    while not self.complete():
      
      for i in range(self.iterations_per):
        new_config = self.neighbour_func(self.cur_config)
        new_val    = self.eval_func(new_config)

        if self.compare_func(new_val, self.cur_max):
          self.max_config = new_config
          self.cur_max    = new_val

        if self.compare_func(new_val, self.cur_value):
          self.cur_config = new_config
          self.cur_value  = new_val
        
        if self.accept_bad_move(new_val, self.cur_value):
          self.cur_config = new_config
          self.cur_value  = new_val
      
      self.decrement_func(self.temp, self.alpha)