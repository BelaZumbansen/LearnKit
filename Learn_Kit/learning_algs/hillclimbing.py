import numpy as np
import math

def maximize_comparison(a, b):
  return a > b
  

class HillClimbing:
  def __init__(self, initial_config, eval_func, neighbour_func, max_iterations=100, compare_func=maximize_comparison):
    self.cur_config     = initial_config
    self.eval_func      = eval_func
    self.neighbour_func = neighbour_func
    self.eval_func_vec  = np.vectorize(eval_func)
    self.compare_func   = compare_func
    self.max_iterations = max_iterations
    self.cur_value      = self.initial_value

  def get_final_result(self):
    return self.cur_value, self.cur_config
  
  def run(self):

    iteration = 0
    while iteration < self.max_iterations:
      neighbour_ls = self.neighbour_func(self.cur_config)
      neighbour_arr = np.array(neighbour_ls)
      neighbour_val_arr = self.eval_func_vec(neighbour_arr)

      max_index = np.argmax(neighbour_val_arr)
      max_val = neighbour_val_arr[max_index]

      if not self.compare_func(max_val, self.cur_value):
        # Our current configuration is a maximum
        break
      else:
        # take a greedy step
        self.cur_config = neighbour_arr[max_index]
        self.cur_value  = max_val
      
      iteration += 1