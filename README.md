# Dropout
DropOut Model for Adherence vs. Non-Adherence Prediction Code 
---
## Baseline
sas/python/handover/handover/dropoutmodel.py (scripte log -- log file für baseline )

## parameter -  segment product 
parameter 

## Classes: 
1. preprocessing/preprocessing.py : 
    DataLoader
    Processor: Preparation to get Customer specific Features Feature Engineering  - write and save data
2. Training: Splitter 
3. Testing 
    Test parameters --segment=="140:Diabetes"


##Pipeline: 
python preprocessing.py --segment "140:Diabetes" --lessfeats /vs. --morefeats
python training.py --model XGBoost --tinyset /vs.large 
python testing.py --report 

