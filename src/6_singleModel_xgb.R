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

dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "target"), with = F]), label = dt.train$target)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "target"), with = F]), label = dt.valid$target)
x.test <- data.matrix(dt.test[, !c("ID", "target"), with = F])

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
## folding
k <- 10
folds <- createFolds(dt.train$target, k = k, list = F)

## feature names
cols.features.all <- names(dt.train[, !c("ID", "target"), with = F])
cols.features.raw <- cols.features.all[! cols.features.all %in% c(cols.basicStats, cols.zero, cols.analysis)]
cols.features.extra <- c(cols.basicStats, cols.zero, cols.analysis)

cols.features <- list(cols.features.raw
                      , unique(c(cols.features.raw, cols.features.extra[1]))
                      , unique(c(cols.features.raw, cols.features.extra[2]))
                      , unique(c(cols.features.raw, cols.features.extra[3]))
                      , unique(c(cols.features.raw, cols.features.extra[1], cols.features.extra[2]))
                      , unique(c(cols.features.raw, cols.features.extra[1], cols.features.extra[3]))
                      , unique(c(cols.features.raw, cols.features.extra[2], cols.features.extra[3]))
                      , cols.features.all
)

## store the result
dt.result <- as.data.table(matrix(rep(0, 4 * k * length(cols.features)), k * length(cols.features)))
setnames(dt.result, c("cols", "cv_num", "result.dval", "resul.valid"))
df.result <- as.data.frame(dt.result) # df is easier to insert rows

dt.summary <- as.data.table(matrix(rep(0, 9 * length(cols.features)), length(cols.features)))
setnames(dt.summary, c("cols", "mean.dval", "max.dval", "min.dval", "sd.dval", "mean.valid", "max.valid", "min.vaild", "sd.valid"))
df.summary <- as.data.frame(dt.summary) # df is easier to insert rows

## cv
for(j in 1:length(cols.features)){
    # store the result
    vec.result.dval <- rep(0, k)
    vec.result.valid <- rep(0, k)
    for(i in 1:k){
        f <- folds == i
        dval <- xgb.DMatrix(data = data.matrix(dt.train[f, cols.features[[j]], with = F]), label = dt.train[f]$target)
        dtrain <- xgb.DMatrix(data = data.matrix(dt.train[!f, cols.features[[j]], with = F]), label = dt.train[!f]$target)
        watchlist <- list(val = dval, train = dtrain)
        
        params <- list(booster = "gbtree"
                       , nthread = 8
                       , eta = .1
                       , min_child_weight = 5
                       , max_depth = 11
                       , subsample = .3
                       , colsample_bytree = 1
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
        
        pred.dval <- predict(clf, dval)
        result.dval <- logLoss(getinfo(dval, "label"), pred.dval)
        vec.result.dval[i] <- result.dval
        
        dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, cols.features[[j]], with = F]), label = dt.valid$target)
        pred.valid <- predict(clf, dmx.valid)
        result.valid <- logLoss(getinfo(dmx.valid, "label"), pred.valid)
        vec.result.valid[i] <- result.valid
        
        df.result[(j - 1) * k + i, ] <- c(j, i, result.dval, result.valid)
    }
    df.summary[j, ] <- c(j, mean(vec.result.dval), max(vec.result.dval), min(vec.result.dval), sd(vec.result.dval)
                         , mean(vec.result.valid), max(vec.result.valid), min(vec.result.valid), sd(vec.result.valid))
}
# max_depth = 11
hist(c(0.4673935, 0.4647636, 0.4691213, 0.4633835, 0.4649215, 0.4588940, 0.4750696, 0.4711140, 0.4773479, 0.4664173))

df.summary
# raw features
# vec.result.dval
# [1] 0.4534090, 0.4620920, 0.4709736, 0.4688558, 0.4669077, 0.4755299, 0.4660341, 0.4706279,0.4638377, 0.4706432
# vec.result.valid
# [1] 0.4684867, 0.4700887, 0.4694368, 0.4682908, 0.4679137, 0.4695359, 0.4694364, 0.4683410, 0.4678962 0.4682889

