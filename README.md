# BNP_Paribas_Cardif_Claims_Management
A Kaggle competition

## 1.0 TODO
1. [29/02/2016 - **not done yet**] funModeling::cross_plot to see variable relavance
2. [29/02/2016 - **Done**] try value counts as the categorical value
3. [02/03/2016 - **not done yet**] understand and implement Laurae's post
3. [06/03/2016 - **not done yet**] find a most important feature for sampling
4. [11/03/2016 - **not done yet**] add noise

## 2.0 GREAT STUFF
### STEPS TO TUNE XGB (https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19083/best-practices-for-parameter-tuning-on-models)
for xgboost here is my steps, usually i can reach almost good parameters in a few steps,  
initialize parameters such: eta = 0.1, depth= 10, subsample=1.0, min_child_weight = 5, col_sample_bytree = 0.2 (depends on feature size), set proper objective for the problem (reg:linear, reg:logistic or count:poisson for regression, binary:logistic or rank:pairwise for classification)  
split %20 for validation, and prepare a watchlist for train and test set, set num_round too high such as 1000000 so you can see the valid prediction for any round value, if at some point test prediction error rises you can terminate the program running,  
1. play to tune depth parameter, generally depth parameter is invariant to other parameters, i start from 10 after watching best error rate for initial parameters then i can compare the result for different parameters, change it 8, if error is higher then you can try 12 next time, if for 12 error is lower than 10 , so you can try 15 next time, if error is lower for 8 you would try 5 and so on.  
2. after finding best depth parameter, i tune for subsample parameter, i started from 1.0 then change it to 0.8 if error is higher then try 0.9 if still error is higher then i use 1.0, and so on.  
3. in this step i tune for min child_weight, same approach above,  
4. then i tune for col_Sample_bytree  
5. now i descrease the eta to 0.05, and leave program running then get the optimum num_round (where error rate start to increase in watchlist progress),  
after these step you can get roughly good parameters (i dont claim best ones), then you can play around these parameters.  
hope it helps  

### STEPS TO DO CORRESPONENCE AND CHAIN ANALYSIS (https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19240/analysis-of-duplicate-variables-correlated-variables-large-post)