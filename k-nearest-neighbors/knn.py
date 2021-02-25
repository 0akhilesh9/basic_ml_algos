import argparse
from random import seed
from random import randrange
from csv import reader
from math import sqrt

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

# Split a dataset into k folds
def cross_validation_split(inp_dataset, fold_count):
	return_splits = list()
	# Perform splitting over a copy of dataset
	dataset_copy = list(inp_dataset)
	split_size = int(len(inp_dataset) / fold_count)
	# Iterate over n-folds
	for i in range(fold_count):
		split = list()
		for tmp_count in range(split_size):
			# Randomly select a index
			index = randrange(len(dataset_copy))
			split.append(dataset_copy.pop(index))
		return_splits.append(split)
	# return the datasplit
	return return_splits

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
	for i in range(len(data1)-1):
		distance_value += (data1[i] - data2[i])**2
	distance_value = sqrt(distance_value)
	return distance_value

# Find the closest 'n' neighbors based on the specified distance metric
def get_neighbors(train_data, test_sample, num_neighbors):
	neighbour_distances = list()
	for train_sample in train_data:
		# Individual distance calculation
		distance_val = distance_euclidean(test_sample, train_sample)
		neighbour_distances.append((train_sample, distance_val))
	neighbour_distances.sort(key=lambda x: x[1])

	return [x[0] for x in neighbour_distances[:5]]

# k nearest neighbors algorithm
def knn_algo(train_data, test_data, num_neighbors):
	predicted_values = list()
	for test_row in test_data:
		# Get nearest 'n' neighbours
		nearest_neighbors = get_neighbors(train_data, test_row, num_neighbors)
		# Neighour labels
		neighbour_values = [x[-1] for x in nearest_neighbors]
		# Max count label
		predicted = max(set(neighbour_values), key=neighbour_values.count)
		predicted_values.append(predicted)
	return(predicted_values)

# Driver program
def train_predict(dataset, n_folds, num_neighours):
	# k-fold Cross validation
	splits = cross_validation_split(dataset, n_folds)
	accuracy_values = list()
	erm_values =[]
	# Iterate over k folds
	for split in splits:
		train_set = list(splits)
		train_set.remove(split)
		train_set = sum(train_set, [])
		test_set = []
		for row in split:
			row_copy = list(row)
			test_set.append(row_copy)
		# Get predicted values
		predicted_value = knn_algo(train_set, test_set, num_neighours)
		actual_value = [x[-1] for x in split]
		# Performance metric calculation
		erm_values.append(calculate_erm(actual_value, predicted_value))
		accuracy_values.append(calculate_accuracy(actual_value, predicted_value))

	return accuracy_values, erm_values

# Driver program
def train_predict_basic(train_set, test_set, n_folds, num_neighours):
	# k-fold Cross validation
	splits = cross_validation_split(dataset, n_folds)
	accuracy_values = list()
	erm_values =[]

	# Get predicted values
	predicted_value = knn_algo(train_set, test_set, num_neighours)
	actual_value = [x[-1] for x in test_set]
	# Performance metric calculation
	erm_values.append(calculate_erm(actual_value, predicted_value))
	accuracy_values.append(calculate_accuracy(actual_value, predicted_value))

	return accuracy_values, erm_values

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


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--k', help='No: of neighbours to consider', default=5, required=False)
parser.add_argument('--mode', help='cross_valid or normal', default="normal", required=False)
parser.add_argument('--nfold', help='N folds for cross validation', default=5, required=False)
args = vars(parser.parse_args())

# Initialize parameters
data_file = args['dataset']
num_neighbours = int(args["k"])
n_folds = int(args["nfold"])
mode = args["mode"]

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
if mode == "cross_valid":
	print("N folds: %d  Num neighbours: %d"%(n_folds, num_neighbours))
	accuracy_scores, erm_values = train_predict(dataset, n_folds, num_neighbours)
	print('Error values: %s' % erm_values)
	print('Mean Error value: %.4f%%' % (sum(erm_values)/float(len(erm_values))))
	print('Accuracy Scores: %s' % accuracy_scores)
	print('Mean Accuracy Score: %.4f%%' % (sum(accuracy_scores)/float(len(accuracy_scores))))

else:
	train_data, test_data = test_train_split(dataset)
	print("Num neighbours: %d"%num_neighbours)
	accuracy_scores, erm_values = train_predict_basic(train_data, test_data, n_folds, num_neighbours)
	print('Error values: %s' % erm_values)
	print('Accuracy Scores: %s' % accuracy_scores)