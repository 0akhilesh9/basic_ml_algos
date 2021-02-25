Cluster metrics - I: 
The percentage of 0 labels and 1 labels within each cluster ex: num of 0 label in cluster i / total num of samples in cluster i
Cluster metrics - II: 
The percentage of all 0 labels and 1 labels ex: num of 0 label in cluster i/ total num of 0 labels in the whole dataset


Command line arguments:

"--dataset" : Path to the csv file  (required)
optional arguments:
  -h, --help            show this help message and exit
  --k K                 No: of neighbours to consider
  --distance            manhattan or chebyshev, minkowski, average and default is eculedian
  --numIter             No: of iterations   (By default 3)
  --maxUpdateIter       No: of max iterations for updating the centroids
  --split_test_train
  --inter_cluster_variance


example run:

python kmeans.py --dataset="Breast_cancer_data.csv"