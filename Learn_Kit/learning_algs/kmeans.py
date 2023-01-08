import numpy as np

def dist_array_to_point(X, point):

  dimensions = np.shape(X)
  dist = np.zeros(dimensions[0])

  Xtemp = X - point

  Xtemp **= 2

  dist = np.sum(Xtemp, axis=1)
  dist = dist**(0.5)

  return dist

class KMeans:
  def __init__(self, data, k, start_means=np.array([]), max_iterations=50):
    self.data = data
    # Let m be the number of data points
    self.m = data.shape[0]
    # Let n be the dimension
    self.n = data.shape[1]
    self.k = k
    self.max_iterations = max_iterations

    self.means = np.array([])
    #Randomly choose initial cluster means unless otherwise specified
    if start_means.size == 0:
      # Randomly choose k indices of the data array M
      random_indices = np.random.choice(self.m,size=k,replace=False)
      # Select the k means as initial means
      self.means = data[random_indices,:]
    else:
      self.means = start_means

    # Initialize arrays of zeros for the distances, labels, and losses
    self.dist = np.zeros((self.k, self.m))
    self.cluster_labels = np.zeros((self.m,))
    self.losses = np.zeros(self.max_iterations)

  def complete(self, num_changed, iteration):
    return num_changed == 0 or iteration > self.max_iterations

  def run(self):

    num_changed = 1
    iteration   = 0

    while not self.complete(num_changed, iteration):

      # Track labels
      cluster_labels_cpy = self.cluster_labels.copy()

      for i in range(self.k):

        # Determine distance to mean i
        self.dist[i,:] = dist_array_to_point(self.data, self.means[i,:])
      
      # Update labels
      self.cluster_labels = np.argmin(self.dist, axis=0)

      # Determine losses for this iteration
      self.losses[iteration] = np.mean(np.min(self.dist, axis=0))

      num_changed = np.sum(self.cluster_labels != cluster_labels_cpy)

      for i in range(self.k):
        points_in_i = self.data[self.cluster_labels==i,:]
        self.means[i,:] = np.mean(points_in_i, axis=0)

      iteration += 1