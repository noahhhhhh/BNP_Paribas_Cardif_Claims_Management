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
## 3.0 train, valid, test #############################################################
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

## feature engineer on train and valid ################################################
## FactorRankSum
# dt.factorRank <- ConvertNonNumFactorToOrderedNum(dt.train, cols.factor[! cols.factor %in% cols.newFeatures])
# dt.factorRank <- dt.factorRank[, lapply(.SD, as.numeric)]
# pre.factorRank <- preProcess(dt.factorRank
#                              , method = c("range")
#                              , verbose = T)
# dt.factorRank.range <- predict(pre.factorRank, dt.factorRank)
# vFactorRankSum <- rowSums(dt.factorRank.range)

## encode
# columns need encoding
cols.needEncode <- names(ColUnique(dt.train[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]))[ColUnique(dt.train[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]) > 10]
# factor encode
ls.encode.factor <- ConvertNonNumFactorToOrderedNum(dt.train, dt.valid, cols.needEncode)
dt.encode.factor.train <- ls.encode.factor[[1]]
dt.encode.factor.valid <- ls.encode.factor[[2]]
setnames(dt.encode.factor.train, names(dt.encode.factor.train), paste(namesd(dt.encode.factor.train), "_factor", sep = ""))
setnames(dt.encode.factor.valid, names(dt.encode.factor.valid), paste(namesd(dt.encode.factor.valid), "_factor", sep = ""))

# numeric encode
ls.encode.numeric <- ConvertNonNumFactorToOrderedNum(dt.train, dt.valid, cols.needEncode)
dt.encode.numeric.train <- ls.encode.numeric[[1]]
dt.encode.numeric.valid <- ls.encode.numeric[[2]]
dt.encode.numeric.train <- dt.encode.numeric.train[, lapply(.SD, as.numeric)]
dt.encode.numeric.valid <- dt.encode.numeric.valid[, lapply(.SD, as.numeric)]
setnames(dt.encode.numeric.train, names(dt.encode.numeric.train), paste(names(dt.encode.numeric.train), "_numeric", sep = ""))
setnames(dt.encode.numeric.valid, names(dt.encode.numeric.valid), paste(names(dt.encode.numeric.valid), "_numeric", sep = ""))

# remove original vars and cbind the encoded vars
dt.train <- dt.train[, names(dt.train)[!names(dt.train) %in% cols.needEncode], with = F]
dt.valid <- dt.valid[, names(dt.valid)[!names(dt.valid) %in% cols.needEncode], with = F]
cols.factor <- cols.factor[!cols.factor %in% cols.needEncode]
dt.train <- cbind(dt.train, dt.encode.factor.train, dt.encode.numeric.train)
dt.valid <- cbind(dt.valid, dt.encode.factor.valid, dt.encode.numeric.valid)
cols.factor <- c(cols.factor, names(dt.encode.factor.train))
cols.numeric <- c(cols.numeric, names(dt.encode.numeric.train))
dim(dt.train); dim(dt.valid)
#######################################################################################

dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "target"), with = F]), label = dt.train$target)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "target"), with = F]), label = dt.valid$target)
x.test <- data.matrix(dt.test[, !c("ID", "target"), with = F])

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
cols.features <- names(dt.train)[!names(dt.train) %in% c("ID", "target")]
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "logloss"
               , md = 9
               , ss = .9
               , mcw = 1
               , csb = .5
               , eta = .025)
df.summary <- myCV_xgb(dt.train
                       , cols.features
                       , dt.valid
                       , k = 0
                       , params)
df.summary
# 0.4633959 benchmark
# 0.4631082 no basic stats, but vFactorRankSum
# 0.4654398 remvoe the wrong encoded features, FactorRankSum, and no basic stats
save(df.summary, file = "data/cv_results/single_xgb_all_features.RData")
