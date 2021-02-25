Command line arguments:

"--dataset" : Path to the csv file  (required)
"--k": No: of neighbours to consider    (not required); default = 5
"--nfold": N folds for cross validation  (not required; default = 5)
"--mode": Normal test-train split or using shuffling -values=["cross_valid","normal"]  (not required, default config is normal test-train split)

example run:

python knn.py --dataset="Breast_cancer_data.csv"