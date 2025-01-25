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

## Target-based drug Prediction:
1)Create a test file named "ad_pre.txt" and move the file to the folder "test_data".

2)Run the following command, predicting results will be saved in the file "pre_results.txt".
```python
  python test.py -data test_data -gpu 1 -name test_model -save_result pre_results.txt -test_file ad_pre.txt
```

### Parameter Note:

-data : the directory of training and testing data

-gpu : the GPU to use

-name : the name of the model snapshot (used for storing model parameters)

-epoch : the number of epochs

-save_result : the filename that is used to store test results

-test_file : the name of testing file


# The ranking of potential drug candidates for diabetes cataract

In the 'DrugRank' folder, We list the top 1000 drug candidates repurposed for potentially reducing the risk of cataract extraction in patients with diabetes mellitus.

# Reference
[1] Zhenxiang Gao, Maria Gorenflo, David Kaelber, Vincent Monnier and Rong Xu. “Drug repurposing for reducing the risk of cataract extraction in patients with diabetes mellitus: integration of artificial intelligence-based drug prediction and clinical corroboration.” Frontiers in Pharmacology, 14, 1181711, 2023.

[2] Zhenxiang Gao, Pingjian Ding, Rong Xu. “KG-Predict: A knowledge graph computational framework for drug repurposing.” Journal of Biomedical Informatics, 132, 104133, 2022.
