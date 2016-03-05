setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(xgboost)
require(Ckmeans.1d.dp)
require(ggplot2)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_preprocessed.RData")
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed[target >= 0]$target, p = .9, list = F) # remember to change it to .66
dt.train <- dt.preprocessed[target >= 0][ind.train]
dt.valid <- dt.preprocessed[target >= 0][-ind.train]
dt.test <- dt.preprocessed[target == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$target)
table(dt.valid$target)

#######################################################################################
## 2.0 cross validate #################################################################
#######################################################################################
k <- 10
folds <- createFolds(dt.train$target, k = k, list = F)
vec.result <- rep(0, k)
for(i in 1:k){
    f <- folds == i
    dval <- xgb.DMatrix(data = data.matrix(dt.train[f, !c("ID", "target"), with = F]), label = dt.train[f]$target)
    dtrain <- xgb.DMatrix(data = data.matrix(dt.train[!f, !c("ID", "target"), with = F]), label = dt.train[!f]$target)
    watchlist <- list(val = dval, train = dtrain)
    
    params <- list(booster = "gbtree"
                    , nthread = 8
                   , eta = .1
                   , min_child_weight = 5
                   , max_depth = 12
                   , subsample = 1
                   , colsample_bytree = .2
                   , objective = "binary:logistic"
                   , eval_metric = "logloss"
                   )
    
    set.seed(888)
    print(paste("cv:", i, "-------"))
    clf <- xgb.train(params = params
                     , data = dtrain
                     , nrounds = 100000 
                     , early.stop.round = 50
                     , watchlist = watchlist
                     , print.every.n = 10
                     )
    
    pred.valid <- predict(clf, dval)
    result <- logLoss(getinfo(dval, "label"), pred.valid)
    vec.result[i] <- result
}

vec.result.raw <- vec.result # 11


