import argparse
from random import seed
from random import randrange
from csv import reader
from math import *
from decimal import Decimal
import matplotlib
import matplotlib.pyplot as plt

# Emperical risk minimization calculation
def calculate_erm(actual, hypothesis):
	predictions = []
	# Iterate over predictions
	for i in range(len(actual)):
		# Get the error in prediction
		predictions.append(float(abs(actual[i] - hypothesis[i])))
	return sum(predictions)/len(predictions)

# Split a dataset into k folds
def test_train_split(inp_dataset, test_percent=0.2):
	return_splits = list()
	test_count = int(test_percent * len(inp_dataset))
	train_count = len(inp_dataset) - test_count
	test_set = []
	dataset_copy = list(inp_dataset)
	# Perform splitting
	for i in range(test_count):
		index = randrange(len(dataset_copy))
		test_set.append(dataset_copy.pop(index))
	# return the datasplit
	return dataset_copy, test_set

# Calculate variance for a cluster
def get_cluster_variance(dataset, centroid_map, cluster_index, centroid_list):
	centroid_indexes = [j for j in range(len(dataset)) if centroid_map[j] == cluster_index]
	cluster_points = [dataset[i] for i in centroid_indexes]
	variance = 0.0
	for j in range(len(cluster_points)):
		for i in range(len(cluster_points[j])):
			variance += (cluster_points[j][i] - centroid_list[cluster_index][i]) ** 2
	return variance

# Calculate overall variance for clusters
def get_total_cluster_variance(dataset, centroid_list):
	mean = [0 for x in range(len(dataset[0]))]
	for i in range(len(dataset)):
		mean = [sum(x) for x in zip(dataset[i], mean)]
	mean = [x / len(dataset) for x in mean]
	variance = 0.0
	for j in range(len(centroid_list)):
		for i in range(len(mean)):
			variance += (mean[i] - centroid_list[j][i]) ** 2
	return variance

# Calculate variance value
def calculate_variance(dataset, centroid_map, centroid_list, inter_cluster=False):
	variance = 0.0
	for cluster_index in range (len(centroid_list)):
		variance += get_cluster_variance(dataset, centroid_map, cluster_index, centroid_list)
	if inter_cluster:
		variance += get_total_cluster_variance(dataset, centroid_list)
	return variance

# Calculate total distance
def calculate_total_distance(dataset, distance_fn, centroid_list, centroid_map):
	sum_dist = 0.0
	for i in range(len(dataset)):
		sum_dist += distance_fn(dataset[i], centroid_list[centroid_map[i]])
	return sum_dist

# Calculate accuracy percentage
def calculate_accuracy(actual, predicted):
	correct_count = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct_count += 1.0
	return correct_count / len(actual)

# Calculate the distance between two samples (Euclidean)
def distance_euclidean(data1, data2):
	# sqrt((x_2-x_1)^2 + (y_2-y_1)^2) for 2 dimensions
	distance_value = 0.0
	for i in range(len(data1)):
		distance_value += (data1[i] - data2[i])**2
	distance_value = sqrt(distance_value)
	return distance_value

# Calculate averaged Euclidean
def distance_average(data1, data2):
	# sqrt(1/n * (x_2-x_1)^2 + (y_2-y_1)^2) for 2 dimensions
	distance_value = 0.0
	for i in range(len(data1)):
		distance_value += (data1[i] - data2[i])**2
	distance_value = sqrt(distance_value / len(data1))
	return distance_value

# Pth root
def pth_root(x, p):
   root_val = 1 / float(p)
   return float(round (Decimal(x) ** Decimal(root_val), 3))

# Minkowski distance
def distance_minkowski(data1, data2, p=3):
   distance_value = (pth_root(sum(pow(abs(a-b), p)  for a, b in zip(data1, data2)), p))
   return distance_value

# Manhattan distance
def distance_manhattan(data1, data2):
	distance_value = 0.0
	for i in range(len(data1)):
		distance_value += abs(data1[i] - data2[i])
	return distance_value

# Chebyshev distance
def distance_chebyshev(data1, data2):
	distance_value = []
	for i in range(len(data1)):
		distance_value.append(abs(data1[i] - data2[i]))
	return max(distance_value)

# Calculate mean of data points
def calculate_mean(datapoints):
	mean_val = []
	for i in range(len(datapoints[0])):
		mean_val.append(sum([x[i] for x in datapoints])/len(datapoints))
	return mean_val

# Visualization plot
def visualize_plot(dataset, centroid_list):
	for centroid in centroid_list:
		plt.scatter([centroid][0], centroid[1], s=130, marker="x")

# Driver program
def train(dataset, k_val, dist_fn, max_update_iter, tolerance=1):
	# Random initialization of centroids
	centroid_list = []
	total_dist = []
	for i in range(k_val):
		index = randrange(len(dataset))
		centroid_list.append(dataset[index])
	centroid_map = []
	for iter in range(max_update_iter):
		# Calculate the distances from the centroids
		for i in range(len(dataset)):
			centroid_map.append(k_val-1)
			tmp_val = float("inf")
			for j in range(k_val):
				if tmp_val > dist_fn(dataset[i], centroid_list[j]):
					tmp_val = dist_fn(dataset[i], centroid_list[j])
					centroid_map[i] = j

		total_dist.append(calculate_total_distance(dataset, dist_fn, centroid_list, centroid_map))

		# Update Centroids
		optimized_flag = True
		for i in range(k_val):
			centroid_indexes = [j for j in range(len(dataset)) if centroid_map[j]==i]
			tmp_val = calculate_mean([dataset[i] for i in centroid_indexes])
			# Convergence condition: if the new centroid value and the old centroid value are different
			if dist_fn(centroid_list[i], tmp_val) > tolerance:
				centroid_list[i] = tmp_val
				optimized_flag = False
		if optimized_flag:
			variance = calculate_variance(dataset, centroid_map, centroid_list, args["inter_cluster_variance"])
			centroid_metrics = predict(dataset, centroid_list, dist_fn)
			break
	return centroid_list, total_dist, variance

# Prediction
def predict(dataset, centroid_list, dist_fn):
	centroid_map = []
	k_val = len(centroid_list)
	# Calculate the distances from the centroids
	for i in range(len(dataset)):
		centroid_map.append(k_val - 1)
		tmp_val = float("inf")
		for j in range(k_val):
			if tmp_val > dist_fn(dataset[i], centroid_list[j]):
				tmp_val = dist_fn(dataset[i], centroid_list[j])
				centroid_map[i] = j
	# Metric calculation
	centroid_metrics = []
	for i in range(k_val):
		centroid_indexes = [j for j in range(len(dataset)) if centroid_map[j] == i]
		postive_count = len([dataset[i] for i in centroid_indexes if dataset[i][-1]==1])
		# Contains +ve and -ve ratios for inter cluster and whole dataset
		cluster_size = len(centroid_indexes)
		centroid_metrics.append([[postive_count/cluster_size, 1 - postive_count/cluster_size, cluster_size, postive_count],
								 [postive_count/len(dataset), (cluster_size - postive_count)/len(dataset), cluster_size, postive_count]])

	return centroid_metrics


# Load the input CSV file
def load_csv(inp_file):
    dataset = []
    # Open the csv file
    with open(inp_file, 'r') as file:
        csv_reader = reader(file)
        # Iterate over all the rows
        for data_row in csv_reader:
            dataset.append(data_row)
    return dataset

# Driver program
def run_prg(args, dist_fn):
	global dataset, k_val, num_iter, debug_flag
	train_data = dataset
	test_data = dataset
	max_update_iter = int(args["maxUpdateIter"])
	if args["split_test_train"]:
		train_data, test_data =  test_train_split(dataset)
	tmp_cluster_list = []
	for i in range(num_iter):
		centroid_list, total_dist, variance = train(train_data, k_val, dist_fn, max_update_iter)
		tmp_cluster_list.append([variance, centroid_list])
	# Select cluster centroids with least variance
	tmp_cluster_list.sort(key=lambda x: x[0])
	centroid_list = tmp_cluster_list[0][1]
	# Stats
	print("Performance metrics for the best found cluster set:")
	centroid_metrics = predict(test_data, centroid_list, dist_fn)
	print("Num clusters: %d" % k_val)
	print("Cluster Metrics - I (Please refer to the readme.txt file")
	for i in range(k_val):
		print("Cluster-%d"%i)
		print("Datapoints: %d"%centroid_metrics[i][0][2])
		print("Percentage of cluster with Positive diagnosis: %f"%(centroid_metrics[i][0][0]*100))
		print("Percentage of cluster with Negative diagnosis: %f" %(centroid_metrics[i][0][1]*100))
		print("Positive label count: %d"%centroid_metrics[i][0][3])

	print("Cluster Metrics - II (Please refer to the readme.txt file")
	for i in range(k_val):
		print("Cluster-%d" % i)
		print("Datapoints: %d" % centroid_metrics[i][0][2])
		print("Percentage of cluster with Positive diagnosis: %f" % (centroid_metrics[i][1][0] * 100))
		print("Percentage of cluster with Negative diagnosis: %f" % (centroid_metrics[i][1][1] * 100))
		print("Positive label count: %d" % centroid_metrics[i][0][3])

	# visualize_plot(dataset, centroid_list)

	if debug_flag:
		print("Centroids: ")
		print(centroid_list)
	return total_dist

# data_file = r"D:\workspace\stonybrook\sem2\machine_learning\bed1\assignment_4\Breast_cancer_data.csv"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--k', help='No: of neighbours to consider', default=2, required=False)
parser.add_argument('--distance', help='cross_valid or normal', default="average", required=False)
parser.add_argument('--numIter', help='No: of iterations', default=3, required=False)
parser.add_argument('--maxUpdateIter', help='No: of max iterations for updating the centroids', default=1000, required=False)

parser.add_argument('--split_test_train', dest='split_test_train', action='store_true')
parser.add_argument('--inter_cluster_variance', dest='inter_cluster_variance', action='store_true')
args = vars(parser.parse_args())

# Initialize parameters
debug_flag = False
data_file = args['dataset']
k_val = int(args["k"])
num_iter = int(args["numIter"])
distance_method = args["distance"]

seed(10)
# Load the input CSV file
dataset = load_csv(data_file)

# First row is a string with parameter names
dataset[0] = [x.strip() for x in dataset[0]]
i = 1
for row in dataset[1:]:
    dataset[i] = [float(x.strip()) for x in row]
    i = i + 1
dataset = dataset[1:]


seed(10)
if distance_method == "manhattan":
	total_dist = run_prg(args, distance_manhattan)

elif distance_method == "chebyshev":
	total_dist = run_prg(args, distance_chebyshev)

elif distance_method == "minkowski":
	total_dist = run_prg(args, distance_minkowski)

elif distance_method == "average":
	total_dist = run_prg(args, distance_average)

else:
	total_dist = run_prg(args, distance_euclidean)

def plot_graphs():
	import time
	total_distance = []
	time_list = []
	start = time.time()
	total_distance.append(["Manhattan", run_prg(distance_manhattan)])
	time_list.append(["Manhattan", time.time()-start])
	start = time.time()
	total_distance.append(["Chebyshev", run_prg(distance_chebyshev)])
	time_list.append(["Chebyshev", time.time() - start])
	start = time.time()
	total_distance.append(["Minkowski", run_prg(distance_minkowski)])
	time_list.append(["Minkowski", time.time() - start])
	start = time.time()
	total_distance.append(["Average euclidean", run_prg(distance_average)])
	time_list.append(["Average euclidean", time.time() - start])
	start = time.time()
	total_distance.append(["Euclidean", run_prg(distance_euclidean)])
	time_list.append(["Euclidean", time.time() - start])

	fig, ax = plt.subplots()
	for j in range(len(total_distance)):
		ax.plot([i for i in range(len(total_distance[j][1]))], total_distance[j][1], label=total_distance[j][0])
	ax.set(xlabel='Iterations', ylabel='Sum Distance', title='Sum distance vs Iterations')
	ax.grid()
	plt.legend(loc='upper right')
	plt.show()

	fig, ax = plt.subplots()
	dist_methods = [x[0] for x in time_list]
	time_taken = [x[1] for x in time_list]
	ax.bar(dist_methods, time_taken)
	ax.set(ylabel='Time taken in seconds', title='Time taken for different distance methods')
	plt.show()

# plot_graphs()
# from sklearn.cluster import KMeans
# Kmean = KMeans(n_clusters=k_val)
# Kmean.fit(dataset)
# centroid_metrics = predict(dataset, Kmean.cluster_centers_, distance_euclidean)
# print("Num clusters: %d"%k_val)
# print('Metric values:')
# print(centroid_metrics)
# print(Kmean.cluster_centers_)