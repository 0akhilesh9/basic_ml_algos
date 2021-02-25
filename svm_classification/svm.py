import random
import argparse
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

# Data generation function
def prepare_data(train_split = 0.8, plot_flag = False):
  # Random sampling
  X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
  # Adding one to x for bias
  x = np.c_[np.ones((X0.shape[0])), X0]
  # Segregating the positive and negative points
  positive_x =[]
  negative_x =[]
  for i,label in enumerate(y):
    if label == 0:
      negative_x.append(x[i])
    else:
      positive_x.append(x[i])
  # Spitting the data into train and test splits
  split_index = int(len(negative_x)*train_split)
  train_data_tmp = []
  test_data_tmp = []
  for i in range(split_index):
    train_data_tmp.append(list(negative_x[i]) + [-1])
    train_data_tmp.append(list(positive_x[i]) + [1])
  for i in range(len(negative_x)-split_index):
    test_data_tmp.append(list(negative_x[split_index + i]) + [-1])
    test_data_tmp.append(list(positive_x[split_index + i]) + [1])

  random.shuffle(train_data_tmp)
  train_data = [np.array([x[:-1] for x in train_data_tmp]), [x[-1] for x in train_data_tmp]]
  test_data = [np.array([x[:-1] for x in test_data_tmp]), [x[-1] for x in test_data_tmp]]

  data_dict = {-1: np.array(negative_x), 1: np.array(positive_x)}

  # Plotting the data
  if(plot_flag):
    plt.scatter([x[1] for x in negative_x], [x[2] for x in negative_x], s=120, marker='_', linewidths=2)
    plt.scatter([x[1] for x in positive_x], [x[2] for x in positive_x], s=120, marker='+', linewidths=2)
    plt.show()

  return train_data, test_data, data_dict

def cost_gradient_calculation(weights, lambd,  x_val, y_val):
  # For Hinge Loss, v_t is either 0 or -(y * x)
  distance = 1 - (y_val * np.dot(x_val, weights))
  dw = np.zeros(len(weights))
  # Miscalssification
  if distance > 0:
    # 2lambda * w_t + v_t
    # dw = (2*lambd * weights) - (x_val * y_val)
    dw[1:] = (2 * lambd * weights[1:]) - (x_val[1:] * y_val)
    dw[0] = - lambd * y_val

    # w_grad = 2 * lambda_val * weights[1:] - x[1:] * y
    # b_grad = - lambda_val * y

  # Correct classification
  else:
    # 2lambda * w_t
    dw[1:] = 2*lambd * weights[1:]

  return dw

# Train SVM
def train_alternate(x_train, y_train, lambda_val, max_epochs = 100):
  weight_vector = []
  weights = np.zeros(x_train.shape[1])
  # stochastic gradient descent
  for epoch in range(1, max_epochs):
    # learning rate
    learning_rate = 1 / (lambda_val * epoch)

    for i in range(len(x_train)):
      # gradient calculation
      grad_value = cost_gradient_calculation(weights, lambda_val, x_train[i], y_train[i])
      # weight update
      weights = weights - (learning_rate * grad_value)
    weight_vector.append(list(weights))

  weights = sum(np.array(weight_vector))/max_epochs

  return weights

# Train SVM
def train(x_train, y_train, lambda_val, max_epochs=100):
  weight_vector = []
  # Initialize
  theta = np.zeros(x_train.shape[1])
  # stochastic gradient descent
  for epoch in range(1, max_epochs):
    # weight update
    weights = (1 / (lambda_val * epoch)) * theta
    for i in range(len(x_train)):
      # gradient calculation
      distance = 1 - (y_train[i] * np.dot(x_train[i], weights))
      # Miscalssification
      if distance > 0:
        theta = theta + (x_train[i] * y_train[i])
      # Correct classification
      else:
        theta = theta

    weight_vector.append(list(weights))
  # weights = sum(np.array(weight_vector))/max_epochs

  return weights

# Test set prediction
def test(x_test, y_test, weights):
  correct_predictions = 0
  for i in range(len(x_test)):
    product = sum(np.array(x_test[i]) * weights)
    # Label assignment based on sign
    pred_class = 1 if product>=0 else -1
    if pred_class == y_test[i]:
      correct_predictions += 1
  print("Total number of points in test set: %d"%len(x_test))
  print("Total points misclassified: %d"%(len(x_test)-correct_predictions))
  #Accuracy calculation
  accuracy = float(correct_predictions) / float(len(x_test))
  return accuracy

# For plotting the maximum-margin hyperplane
def draw(weights, plot_data):
  # Plot the data points
  y_values = [p[1] for p in plot_data]
  plt.scatter([x[0][0] for x in plot_data], [x[0][1] for x in plot_data], c=y_values, s=30, cmap=plt.cm.Paired)

  # plot the maximum-margin hyperplane
  ax = plt.gca()
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()

  # creating grid
  tmp_x_values = np.linspace(x_lim[0], x_lim[1], 30)
  tmp_y_values = np.linspace(y_lim[0], y_lim[1], 30)
  plot_y_values, plot_x_values = np.meshgrid(tmp_y_values, tmp_x_values)
  plot_points = np.vstack([plot_x_values.ravel(), plot_y_values.ravel()]).T
  z_values = np.array([weights[0] + weights[1] * p[0] + weights[2] * p[1] for p in plot_points])
  z_values = z_values.reshape(plot_x_values.shape)

  # plot decision boundary and margins
  ax.contour(plot_x_values, plot_y_values, z_values, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

  # plot support vectors
  l = []
  neg_class_points=[]
  for point in [x for x in plot_data if x[1]==-1]:
    neg_class_points.append([abs(weights[1:].dot(point[0]) + weights[0] + 1), point])
  neg_class_points.sort(key=lambda x: x[0])
  pos_class_points = []
  for point in [x for x in plot_data if x[1] == 1]:
    pos_class_points.append([abs(weights[1:].dot(point[0]) + weights[0]), point])
  pos_class_points.sort(key=lambda x: x[0])

  # +ve class
  for val in pos_class_points:
    # Rounding off to 2 digits
    # if round(val[0], 2) == 1:
    if int(val[0] * 100)/100 == 1:
      l.append(val[1])
  # -ve class
  for val in neg_class_points:
    # Rounding off to 2 digits
    # if round(val[0], 2) == 0:
    if int(val[0] * 100)/100 == 0:
      l.append(val[1])

  ax.scatter([x[0][0] for x in l], [x[0][1] for x in l], s=100, linewidth=1, facecolors='none', edgecolors='k')
  plt.show()

if __name__ == "__main__":
  # Argument parser
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainMode', help='default or alternate', default="default", required=False)
  args = vars(parser.parse_args())

  # get the dataset file
  train_mode = args['trainMode']

  lambda_val = 0.5
  train_data, test_data, data_dict = prepare_data()

  # Training
  if train_mode == "default":
    weights = train(train_data[0], train_data[1], lambda_val, max_epochs = 10000)
  else:
    weights = train_alternate(train_data[0], train_data[1], lambda_val, max_epochs=100000)

  # Testing
  test_accuracy = test(test_data[0], test_data[1], weights)
  print("Test accuracy: %f %%"%(test_accuracy*100))
  print("Weights: ",weights)

  # Graph plotting
  plot_data1 = [[x[1:] for x in train_data[0]], train_data[1]]
  plot_data = []
  for y in data_dict.keys():
    for x in data_dict[y]:
      plot_data.append([x[1:], y])
  draw(weights, plot_data)

