# DualKG-DC
## A Dual-Layer Knowledge Graph Framework for Drug Combination Prediction


## Requirements:
Python(version >= 3.6)
pytorch(version>=1.4.0)
ordered_set(version>=3.1)
numpy(version>=1.16.2)
torch_scatter(version>=2.0.4)
scikit_learn(version>=0.21.1)

We highly recommend you use Conda for package management.


## Model Training:
1)Create a folder "test_data" under folder "data" and move training data, valid data, and test data to the folder. 

2)Use the following command to train the model, the model will be named as "test_model" and saved in the directory "model_saved".
```python
  python main.py -data test_data -gpu 1 -name test_model -epoch 500
```

## Transudative prediction: identifying known drug-drug-disease:
1)Create a test file named "test_transudative.txt" and move the file to the folder "test_data".

2)Run the following command, predicting results will be saved in the file "pre_results.txt".
```python
  python test_transductive.py -data test_data -gpu 1 -name test_data -save_result results_transudative.txt -disease_list disease_list.txt -combination_ids test_transudative.txt
```

## Inductive prediction: identifying novel drug-drug-disease triples:
1)Create a test file named "test_inductive.txt" and move the file to the folder "test_data".

2)Run the following command, predicting results will be saved in the file "pre_results.txt".
```python
  python test_inductive.py.py -data test_data -gpu 1 -name test_data -save_result results_inductive.txt -disease_list disease_list.txt -combination_ids test_inductive.txt
```

### Parameter Note:

-data : the directory of training and testing data

-gpu : the GPU to use

-name : the name of the model snapshot (used for storing model parameters)

-epoch : the number of epochs

-disease_list ： the name of disease file

-save_result : the filename that is used to store test results

-combination_ids : the name of testing file

