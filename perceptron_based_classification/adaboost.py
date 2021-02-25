from random import randrange
from random import seed
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
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

# Plot function
def plot_fig(plot_list, xlabel, ylabel, title, fig_name):
    plt.figure(fig_name)
    # Line plot
    for plot in plot_list:
        plt.plot(plot[0], plot[1], label=plot[2])
    # Set ticks, legends, title, limits
    x_axis = list(set(plot_list[0][0][:-1]))
    y_axis = list(set(plot_list[0][1]))
    # Set different parameters
    plt.xticks(x_axis, rotation=90)
    plt.yticks(y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(ymin=0)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

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

# Accuracy calculation
def accuracy_score(y_actual, y_pred):
    return np.sum(y_actual == y_pred, axis=0) / len(y_pred)

# Split the dataset
def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    # Shuffle the dataset
    np.random.seed(seed)
    idx_val = np.arange(X.shape[0])
    np.random.shuffle(idx_val)
    X = X[idx_val]
    y = y[idx_val]
    # Split the dataset into train and test
    split_index = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

# Decision stump for Adaboost
class ClassDecisionStump():
    def __init__(self):
        # To check if sample to be classified as -1 or 1 given threshold
        self.polarity = 1
        # feature index used for classification
        self.x_feature_index = None
        # threshold value
        self.threshold_val = None
        # classifier's accuracy parameter
        self.alpha_weight = None

class AdaboostClass():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, feature_count = np.shape(X)

        # Initialize weights to 1/N
        sample_weights = np.full(n_samples, (1 / n_samples))
        hypothesis = []
        self.classifiers = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = ClassDecisionStump()
            # Minimum error given for predicting sample label
            min_error = float('inf')
            # Iterate through features
            for feature_i in range(feature_count):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Try every unique feature value as threshold
                for threshold_val in unique_values:
                    tmp_polarity = 1
                    # initial predictions
                    prediction = np.ones(np.shape(y))
                    # Label -1 for samples < threshold
                    prediction[X[:, feature_i] < threshold_val] = -1
                    # Error
                    error = sum(sample_weights[y != prediction])
                    
                    # if error = 0.6,  1 - error = 0.4
                    if error > 0.5:
                        error = 1 - error
                        tmp_polarity = -1

                    # save values
                    if error < min_error:
                        clf.polarity = tmp_polarity
                        clf.threshold_val = threshold_val
                        clf.x_feature_index = feature_i
                        min_error = error
            # alpha calculation
            clf.alpha_weight = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Initial predictions to '1'
            predictions = np.ones(np.shape(y))
            # sample values < threshold
            negative_idx = (clf.polarity * X[:, clf.x_feature_index] < clf.polarity * clf.threshold_val)
            # Label -1
            predictions[negative_idx] = -1
            # Calculate latest weights and normalize
            sample_weights *= np.exp(-clf.alpha_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # Save classifier details
            self.classifiers.append(clf)
            hypothesis.append(clf.alpha_weight)
        
        return hypothesis
    
    # Predict function
    def predict(self, x_val):
        # initialize
        n_samples = np.shape(x_val)[0]
        y_pred = np.zeros((n_samples, 1))
        # Iterate over each classifier 
        for clf in self.classifiers:
            # Set all predictions to one
            predictions = np.ones(np.shape(y_pred))
            # indexes where sample value < threshold
            negative_idx = (clf.polarity * x_val[:, clf.x_feature_index] < clf.polarity * clf.threshold_val)
            # Labelling as '-1'
            predictions[negative_idx] = -1
            # Add predictions weighted by the classifiers weight
            y_pred += clf.alpha_weight * predictions

        # Sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred

# Emperical risk minimization calculation
def calculate_erm(y_pred, y_actual):
    predictions = []
    # Iterate over the entire dataset
    for i in range(len(y_pred)):
        prediction = 0.0 if y_pred[i]==y_actual[i] else 1.0         
        # Get the error in prediction            
        predictions.append(prediction)
    return sum(predictions)/len(predictions)

# Format the feature and label data
def process_dataset(dataset):
    x=[]
    y=[]
    for row in dataset:
        x.append([p for p in row[:-1]])
        y.append(row[-1])
    return x,y

# Train the model
def perform_adaboost(X_train, X_test, y_train, y_test, t_val):
    clf = AdaboostClass(n_clf=t_val)
    # training
    hypothesis = clf.fit(X_train, y_train)
    # prediction
    y_pred = clf.predict(X_test)

    #accuracy = accuracy_score(y_test, y_pred)
    # erm calculation
    error = calculate_erm(y_pred, y_test)
    #print ("Accuracy:", accuracy)
    return hypothesis, error

# Cross validation n_fold
def cross_validation_train(dataset, n_folds, t_val):
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
        # format the dataset
        X_train, y_train = process_dataset(train_set)
        X_test, y_test = process_dataset(test_set)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array([1 if p==1 else -1 for p in y_train])
        y_test = np.array([1 if p==1 else -1 for p in y_test])
        
        # Train
        hypothesis, erm_value = perform_adaboost(X_train, X_test, y_train, y_test, t_val)
        # Error calculation
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

# First row is a string with parameter names
dataset[0] = [x.strip() for x in dataset[0]]
i = 1
for row in dataset[1:]:
    dataset[i] = [float(x.strip()) for x in row]
    i = i + 1
dataset = dataset[1:]

n_folds = args['nfold']
print("No of  folds: %d"%int(args['nfold']))

t_val = 5

if mode=="erm":
    X,y = process_dataset(dataset)
    zx = np.array(X)
    zy = np.array([1 if p==1 else -1 for p in y])
    # Test-Train splitting
    X_train, X_test, y_train, y_test = train_test_split(zx, zy, test_size=0.3)
    # Training and prediction
    hypothesis, erm_value = perform_adaboost(X_train, X_test, y_train, y_test, t_val)
    print("ERM mode")
    print("Hypothesis:")
    print(hypothesis)
    print("Error of prediction:")
    print(erm_value)

elif mode=="cross_valid":  
    erm_values = cross_validation_train(dataset, n_folds, t_val)
    print("Cross validation mode: %d folds"%n_folds)
    print("Error values:")
    print(erm_values)
    print("Average error: %f"%(sum(erm_values)/n_folds))
elif mode=="analysis":
    print("Comparative analysis:")
    X,y = process_dataset(dataset)
    zx = np.array(X)
    zy = np.array([1 if p==1 else -1 for p in y])
    # Test-Train splitting
    X_train, X_test, y_train, y_test = train_test_split(zx, zy, test_size=0.3)
    erm_normal = []
    erm_cross_val = []
    t_max = 7
    for i in range(t_max):
        t_val = i
        print("\n")
        print("T value: %d"%t_val)
        # Training and prediction
        hypothesis, erm_value = perform_adaboost(X_train, X_test, y_train, y_test, t_val)
        erm_normal.append(erm_value)
        erm_values = cross_validation_train(dataset, n_folds, t_val)
        erm_cross_val.append((sum(erm_values)/n_folds))
    
    x_values = list(range(1,t_max+1))
    plot_list = []
    plot_list.append([x_values, erm_normal, "Normal Emperical Risk"])
    plot_list.append([x_values, erm_cross_val, "Cross Validation"])
    plot_fig(plot_list, 'T value', 'ERM', 'Q2_Plot', 'Comparative Analysis of Normal vs Cross validation')
else:
    print("Invalid mode. Select 'erm' or 'cross_valid")


