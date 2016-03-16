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
load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_preprocessed.RData")

#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed[target >= 0]$target, p = .8, list = F) # remember to change it to .66
dt.train <- dt.preprocessed[target >= 0][ind.train]
dt.valid <- dt.preprocessed[target >= 0][-ind.train]
dt.test <- dt.preprocessed[target == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$target)
table(dt.valid$target)

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
require(extraTrees)
mx.train <- model.matrix(target ~., dt.train)
clf.extra <- extraTrees(mx.train
                        , dt.train$target
                        , numThreads = 8)