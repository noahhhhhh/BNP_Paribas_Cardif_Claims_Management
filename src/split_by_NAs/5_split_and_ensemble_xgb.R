setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(xgboost)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(combinat)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_preprocessed.RData")
cols.A.features <- names(dt.group.A)[!names(dt.group.A) %in% c("ID", "target")]
cols.B.features <- names(dt.group.B)[!names(dt.group.B) %in% c("ID", "target")]
cols.C.features <- names(dt.group.C)[!names(dt.group.C) %in% c("ID", "target")]
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.group.A[target >= 0]$target, p = .9, list = F) # remember to change it to .66
dt.train <- dt.group.A[target >= 0][ind.train]
dt.valid <- dt.group.A[target >= 0][-ind.train]
dt.test <- dt.group.A[target == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

dt.valid.B <- dt.group.B[target >= 0][dt.group.B[target >= 0]$ID %in% dt.valid$ID]
dt.valid.C <- dt.group.C[target >= 0][dt.group.C[target >= 0]$ID %in% dt.valid$ID]
dim(dt.valid); dim(dt.valid.B); dim(dt.valid.C)
# [1] 11432    42
# [1] 6531   65
# [1] 6424  127

dt.train.B <- dt.group.B[!dt.group.B$ID %in% dt.valid.B$ID][target >=0]
dt.test.B <- dt.group.B[target <= 0]
dt.train.C <- dt.group.C[!dt.group.C$ID %in% dt.valid.C$ID][target >=0]
dt.test.C <- dt.group.C[target <= 0]
dim(dt.train.B); dim(dt.train.C)
# [1] 59171    65
# [1] 58101   127
dim(dt.test.B); dim(dt.test.C)
# 1] 81950    65
# [1] 80408   127

## group A dmx
dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "target"), with = F]), label = dt.train$target)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "target"), with = F]), label = dt.valid$target)
x.test <- data.matrix(dt.test[, !c("ID", "target"), with = F])

## group B dmx
dmx.train.B <- xgb.DMatrix(data = data.matrix(dt.train.B[, !c("ID", "target"), with = F]), label = dt.train.B$target)
dmx.valid.B <- xgb.DMatrix(data = data.matrix(dt.valid.B[, !c("ID", "target"), with = F]), label = dt.valid.B$target)
x.test.B <- data.matrix(dt.test.B[, !c("ID", "target"), with = F])

## group C dmx
dmx.train.C <- xgb.DMatrix(data = data.matrix(dt.train.C[, !c("ID", "target"), with = F]), label = dt.train.C$target)
dmx.valid.C <- xgb.DMatrix(data = data.matrix(dt.valid.C[, !c("ID", "target"), with = F]), label = dt.valid.C$target)
x.test.C <- data.matrix(dt.test.C[, !c("ID", "target"), with = F])

#######################################################################################
## 2.0 simple ensemble ################################################################
#######################################################################################
## train A
dval <- xgb.DMatrix(data = data.matrix(dt.valid[, cols.A.features, with = F]), label = dt.valid$target)
dtrain <- xgb.DMatrix(data = data.matrix(dt.train[, cols.A.features, with = F]), label = dt.train$target)
watchlist <- list(val = dval, train = dtrain)

params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "logloss"
               , md = 10
               , ss = 1
               , mcw = 0
               , csb = .6
               , eta = .025)
set.seed(888)
clf <- xgb.train(params = params
                 , data = dtrain
                 , nrounds = 100000 
                 , early.stop.round = 50
                 , watchlist = watchlist
                 , print.every.n = 10
)

pred.dval <- predict(clf, dval)
cat("single model on A - valid score:\n")
result.dval <- logLoss(getinfo(dval, "label"), pred.dval)
print(result.dval)
dt.result.A <- data.table(ID = dt.valid$ID, PredictedProb = pred.dval)

## train B
dval.B <- xgb.DMatrix(data = data.matrix(dt.valid.B[, cols.B.features, with = F]), label = dt.valid.B$target)
dtrain.B <- xgb.DMatrix(data = data.matrix(dt.train.B[, cols.B.features, with = F]), label = dt.train.B$target)
watchlist.B <- list(val = dval.B, train = dtrain.B)

params.B <- list(booster = "gbtree"
                 , nthread = 8
                 , objective = "binary:logistic"
                 , eval_metric = "logloss"
                 , md = 9
                 , ss = .9
                 , mcw = 1
                 , csb = .5
                 , eta = .025)
set.seed(888)
clf.B <- xgb.train(params = params.B
                 , data = dtrain.B
                 , nrounds = 100000 
                 , early.stop.round = 50
                 , watchlist = watchlist.B
                 , print.every.n = 10
)

pred.dval.B <- predict(clf.B, dval.B)
cat("single model on A - valid score:\n")
result.dval <- logLoss(getinfo(dval.B, "label"), pred.dval.B)
print(result.dval)
dt.result.B <- data.table(ID = dt.valid.B$ID, PredictedProb = pred.dval.B)

## train C
dval.C <- xgb.DMatrix(data = data.matrix(dt.valid.C[, cols.C.features, with = F]), label = dt.valid.C$target)
dtrain.C <- xgb.DMatrix(data = data.matrix(dt.train.C[, cols.C.features, with = F]), label = dt.train.C$target)
watchlist.C <- list(val = dval.C, train = dtrain.C)

params.C <- list(booster = "gbtree"
                 , nthread = 8
                 , objective = "binary:logistic"
                 , eval_metric = "logloss"
                 , md = 6
                 , ss = .9
                 , mcw = 4
                 , csb = .5
                 , eta = .025)
set.seed(888)
clf.C <- xgb.train(params = params.C
                   , data = dtrain.C
                   , nrounds = 100000 
                   , early.stop.round = 50
                   , watchlist = watchlist.C
                   , print.every.n = 10
)

pred.dval.C <- predict(clf.C, dval.C)
cat("single model on A - valid score:\n")
result.dval <- logLoss(getinfo(dval.C, "label"), pred.dval.C)
print(result.dval)
dt.result.C <- data.table(ID = dt.valid.C$ID, PredictedProb = pred.dval.C)

## ensenmble
logLoss(getinfo(dval, "label"), pred.dval)
# 0.4648545
logLoss(getinfo(dval.B, "label"), pred.dval.B)
# 0.4791578
logLoss(getinfo(dval.C, "label"), pred.dval.C)
# 0.478511
dt.result.A
dt.result.B
dt.result.C
dt.result.ensemble <- merge(merge(dt.result.A, dt.result.B, by = "ID", all.x = T)
                            , dt.result.C, by = "ID", all.x = T)
vec.result.ensemble <- rep(.5, dim(dt.result.ensemble)[1])
for(i in 1:dim(dt.result.ensemble)[1]){
    if(is.na(dt.result.ensemble[i][[3]]) && is.na(dt.result.ensemble[i][[4]])){
        vec.result.ensemble[i] <- dt.result.ensemble[i][[2]]
    } else if(is.na(dt.result.ensemble[i][[3]]) && !is.na(dt.result.ensemble[i][[4]])){
        vec.result.ensemble[i] <- dt.result.ensemble[i][[2]] * .75 + dt.result.ensemble[i][[4]] * .25
    } else if(!is.na(dt.result.ensemble[i][[3]]) && is.na(dt.result.ensemble[i][[4]])){
        vec.result.ensemble[i] <- dt.result.ensemble[i][[2]] * .75 + dt.result.ensemble[i][[3]] * .25
    } else if(!is.na(dt.result.ensemble[i][[3]]) && !is.na(dt.result.ensemble[i][[4]])){
        vec.result.ensemble[i] <- dt.result.ensemble[i][[2]] * .5 + dt.result.ensemble[i][[3]] * .25 + dt.result.ensemble[i][[4]] * .25
    } else {
        vec.result.ensemble[i] <- .5
    }
}

logLoss(getinfo(dval, "label"), vec.result.ensemble)
# 0.463













