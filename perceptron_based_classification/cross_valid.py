import argparse

# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader
 

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
 
# Split a dataset into k folds
def cross_validation_split(dataset, fold_count):
    dataset_split = list()
    # Perform splitting over a copy of dataset
    dataset_copy = list(dataset)
    split_size = int(len(dataset) / fold_count)
    # Iterate over n-folds
    for i in range(fold_count):
        split = list()
        while len(split) < split_size:
            index = randrange(len(dataset_copy))
            split.append(dataset_copy.pop(index))
        dataset_split.append(split)
    # return the datasplit
    return dataset_split
 
# Evaluate an algorithm using a cross validation split
def cross_validation_train(dataset, n_folds, epochs, epoch_flag=False):
    # get n_fold data
    splits = cross_validation_split(dataset, n_folds)
    scores = []
    erm_values = []
    iteration = 0
    # Iterate over the n_fold data
    for split in splits:
        train_set = list(splits)
        train_set.remove(split)
        train_set = sum(train_set, [])
        test_set = []
        for row in split:
            row_copy = list(row)
            test_set.append(row_copy)
        if epoch_flag:
            # Train
            hypothesis = train_algorithm(train_set, epochs)
        else:
            # Train
            hypothesis = train_algorithm_nonepoch(train_set, epochs)
        # Error calculation
        erm_value = calculate_erm(test_set, hypothesis)
        erm_values.append(erm_value)
        print("Split %d details:"%iteration)
        print("Hypothesis:")
        print(hypothesis)
        print("Error of prediction:")
        print(erm_value)
        print("\n")
        iteration = iteration + 1
        # accuracy = accuracy_metric(actual, predicted)
        # scores.append(accuracy)
    return erm_values
 
# Perform training of the perceptron
def train_algorithm(dataset, epochs, l_rate=0.01):
    # Last value in each row of dataset is the label
    # Hypothesis (First value in weights list is bias)
    weights = [0.0 for i in range(len(dataset[0]))]
    sigma_error = 0.0
    # Iterate over all epochs
    for epoch in range(epochs):
        # Iterate over the entire dataset
        for row in dataset:
            activation = weights[0]
            for i in range(len(row)-1):
                activation += weights[i + 1] * row[i]
            prediction = 1.0 if activation >= 0.0 else 0.0         
            # Get the error in prediction            
            error = row[-1] - prediction
            sigma_error = sigma_error + error
            # Update the Hypothesis
            # weights[0] = weights[0] + l_rate * error
            weights[0] = weights[0] + error
            for i in range(len(row)-1):
                #weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                weights[i + 1] = weights[i + 1] + error * row[i]
    return weights

# Perform training of the perceptron
def train_algorithm_nonepoch(dataset, delta=0.001, l_rate=0.01):
    # Last value in each row of dataset is the label
    # Hypothesis (First value in weights list is bias)
    weights = [0.0 for i in range(len(dataset[0]))]
    sigma_error = 9999999.0
    itr = 0
    # Iterate till converges
    while sigma_error/len(dataset) > delta:
        itr = itr + 1
        sigma_error = 0.0
        # Iterate over the entire dataset
        for row in dataset:
            activation = weights[0]
            for i in range(len(row)-1):
                activation += weights[i + 1] * row[i]
            prediction = 1.0 if activation >= 0.0 else 0.0         
            # Get the error in prediction            
            error = row[-1] - prediction
            sigma_error = sigma_error + abs(error)
            # Update the Hypothesis
            # weights[0] = weights[0] + l_rate * error
            weights[0] = weights[0] + error
            for i in range(len(row)-1):
                #weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                weights[i + 1] = weights[i + 1] + error * row[i]
            print("Error is %f"%(sigma_error/len(dataset)))
    print("Converged in %d iterations"%itr)
    return weights

# Emperical risk minimization calculation
def calculate_erm(dataset, hypothesis):
    predictions = []
    # Iterate over the entire dataset
    for row in dataset:
        activation = hypothesis[0]
        for i in range(len(row)-1):
            activation += hypothesis[i + 1] * row[i]
        prediction = 1.0 if activation >= 0.0 else 0.0         
        # Get the error in prediction            
        predictions.append(float(abs(row[-1] - prediction)))
    return sum(predictions)/len(predictions)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--mode', help='erm or cross_valid', required=True)
parser.add_argument('--nfold', help='erm or cross_valid', default=10, required=False)
args = vars(parser.parse_args())

# get the dataset file
data_file = args['dataset']
mode = args["mode"]

seed(1)
# Load the input CSV file
dataset = load_csv(data_file)

n_folds = args['nfold']
print("No of  folds: %d"%int(args['nfold']))

# First row is a string with parameter names
dataset[0] = [x.strip() for x in dataset[0]]
i = 1
for row in dataset[1:]:
    dataset[i] = [float(x.strip()) for x in row]
    i = i + 1
dataset = dataset[1:]

l_rate = 0.01
n_epoch = 500
delta = 0.001

if mode=="erm_epoch":
    hypothesis = train_algorithm(dataset, n_epoch)
    print("ERM mode")
    print("Hypothesis:")
    print(hypothesis)
    erm_value = calculate_erm(dataset, hypothesis)
    print("Error of prediction:")
    print(erm_value)
elif mode=="erm":
    hypothesis = train_algorithm_nonepoch(dataset, delta=delta)
    print("ERM mode")
    print("Hypothesis:")
    print(hypothesis)
    erm_value = calculate_erm(dataset, hypothesis)
    print("Error of prediction:")
    print(erm_value)
elif mode=="cross_valid_epoch":  
    print("Cross validation mode: %d folds"%n_folds)
    erm_values = cross_validation_train(dataset, n_folds, n_epoch, epoch_flag=True)
    print("Error values:")
    print(erm_values)
    print("Average error: %f"%(sum(erm_values)/n_folds))
elif mode=="cross_valid":
    print("Cross validation mode: %d folds"%n_folds)
    erm_values = cross_validation_train(dataset, n_folds, delta, epoch_flag=False)
    print("Error values:")
    print(erm_values)
    print("Average error: %f"%(sum(erm_values)/n_folds))
else:
    print("Invalid mode. Select 'erm' or 'cross_valid' or 'erm_epoch' or 'cross_valid_epoch' ")