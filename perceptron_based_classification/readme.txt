For cross validation default is 10 folds. 

for cross validation
--nfolds = 10 

The program runs till it converges in the below modes:
python cross_valid.py --dataset "Breast_cancer_data.csv" --mode "erm"
python cross_valid.py --dataset "Breast_cancer_data.csv" --mode "cross_valid"

The program runs for 500 epochs in the below modes:
python cross_valid.py --dataset "Breast_cancer_data.csv" --mode "erm_epoch"
python cross_valid.py --dataset "Breast_cancer_data.csv" --mode "cross_valid_epoch"

For programming question 2 (Adaboost):

Select analysis mode for graphical analysis


python adaboost.py --dataset "Breast_cancer_data.csv" --mode "erm"
python adaboost.py --dataset "Breast_cancer_data.csv" --mode "cross_valid"
python adaboost.py --dataset "Breast_cancer_data.csv" --mode "analysis"