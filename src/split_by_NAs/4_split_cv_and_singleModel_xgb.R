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
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
ls.A.TrainValidTest <- myTrainValidTest(dt.group.A)
ls.B.TrainValidTest <- myTrainValidTest(dt.group.B)
ls.C.TrainValidTest <- myTrainValidTest(dt.group.C)

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
## dt.group.A
dt.train.A <- ls.A.TrainValidTest[[1]]
dt.valid.A <- ls.A.TrainValidTest[[2]]
cols.A.features <- names(dt.group.A)[!names(dt.group.A) %in% c("ID", "target")]
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "logloss"
               , md = 10
               , ss = 1
               , mcw = 0
               , csb = .6
               , eta = .025)
df.summary.A <- myCV_xgb(dt.train.A, cols.A.features, dt.valid.A, k = 10, params)
# 0.465115451675942 ss 1, md 6, mcw 4, csb .4, eta .025
# 0.4659392 ss 1, md 5, mcw 4, csb .4, eta .025
# 0.4643789 ss 1, md 8, mcw 4, csb .4, eta .025
# 0.4634195 ss 1, md 10, mcw 4, csb .4, eta .025
# 0.464192 valid (0.4632793) ss 1, md 14, mcw 4, csb .4, eta .025
# 0.4638829 valid (0.462719) ss 1, md 12, mcw 4, csb .4, eta .025
# 0.4639296 valid (0.4626654) ss 1, md 11, mcw 4, csb .4, eta .025
# 0.4637631 valid (0.4631836) ss 1, md 9, mcw 4, csb .4, eta .025
# 0.4638051 valid (0.4625886) ss 1, md 10, mcw 4, csb .4, eta .025
# 0.463987 valid (0.4629434) ss 1, md 8, mcw 4, csb .4, eta .025
# 0.4645659 valid (0.4634756) ss .8, md 10, mcw 4, csb .4, eta .025
# 0.4639391 valid (0.4633465) ss .9, md 10, mcw 4, csb .4, eta .025
# 0.4639337 valid (0.4630845) ss 1, md 10, mcw 6, csb .4, eta .025
# 0.4636882 valid (0.4629213) ss 1, md 10, mcw 5, csb .4, eta .025
# 0.4635482 valid (0.4625394) ss 1, md 10, mcw 3, csb .4, eta .025
# 0.4633697 valid (0.4623626) ss 1, md 10, mcw 2, csb .4, eta .025
# 0.462924 valid(0.46182) ss 1, md 10, mcw 1, csb .4, eta .025
# 0.4632129 valid (0.4621871) ss 1, md 10, mcw 0, csb .4, eta .025
# 0.4632272 valid (0.4624215) ss 1, md 10, mcw 0, csb .6, eta .025 *
# 0.4656532 valid (0.4655925) ss 1, md 10, mcw 0, csb .6, eta .015

# best --------------
# md <- 10
# ss <- 1
# mcw <- 0
# csb <- .6
# eta <- .025
# -------------------

## dt.group.B
dt.train.B <- ls.B.TrainValidTest[[1]]
dt.valid.B <- ls.B.TrainValidTest[[2]]
cols.B.features <- names(dt.group.B)[!names(dt.group.B) %in% c("ID", "target")]
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "logloss"
               , md = 9
               , ss = .9
               , mcw = 1
               , csb = .5
               , eta = .025)
df.summary.B <- myCV_xgb(dt.train.B, cols.B.features, dt.valid.B, k = 10, params)
# 0.4750233 (valid 0.4730986) ss 1, md 10, mcw 0, csb .6, eta .025
# 0.4767156 (valid 0.4732672) ss 1, md 7, mcw 0, csb .6, eta .025
# 0.4791204 (valid 0.4764909) ss 1, md 5, mcw 0, csb .6, eta .025
# 0.4759751 (valid 0.4750073) ss 1, md 12, mcw 0, csb .6, eta .025
# 0.4752849 (valid 0.4735588) ss 1, md 11, mcw 0, csb .6, eta .025
# 0.4754634 (valid 0.4725131) ss 1, md 9, mcw 0, csb .6, eta .025
# 0.4753607 (valid 0.4730938) ss 1, md 9, mcw 1, csb .6, eta .025
# 0.4767099 (valid 0.4744545) ss 1, md 9, mcw 4, csb .6, eta .025
# 0.4759715 (valid 0.4735154) ss 1, md 9, mcw 3, csb .6, eta .025
# 0.4768662 (valid 0.4734225) ss 1, md 9, mcw 2, csb .6, eta .025
# 0.4752358 (valid 0.4725916) ss .8, md 9, mcw 1, csb .6, eta .025
# 0.4771456 (valid 0.4742078) ss .6, md 9, mcw 1, csb .6, eta .025
# 0.4755318 (valid 0.4730891) ss .7, md 9, mcw 1, csb .6, eta .025
# 0.4750495 (valid 0.4723933) ss .9, md 9, mcw 1, csb .6, eta .025
# 0.4748072 (valid 0.4733607) ss .9, md 9, mcw 1, csb .2, eta .025
# 0.4744294 (valid 0.4721211) ss .9, md 9, mcw 1, csb .4, eta .025
# 0.4743646 (valid 0.4721212) ss .9, md 9, mcw 1, csb .5, eta .025 *
# 0.4745565 (valid 0.4713229) ss .9, md 9, mcw 1, csb .5, eta .015
# 0.4753109 (0.4729678) ss .9, md 9, mcw 1, csb .5, eta .035
# best ---------------
# md <- 9
# ss <- .9
# mcw <- 1
# csb <- .5
# eta <- .025
# --------------------

## dt.group.C
dt.train.C <- ls.C.TrainValidTest[[1]]
dt.valid.C <- ls.C.TrainValidTest[[2]]
cols.C.features <- names(dt.group.C)[!names(dt.group.C) %in% c("ID", "target")]
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "logloss"
               , md = 6
               , ss = .9
               , mcw = 4
               , csb = .5
               , eta = .025)
df.summary.C <- myCV_xgb(dt.train.C, cols.C.features, dt.valid.C, k = 10, params)
# 0.476332 (valid 0.4808514) md 9, ss .9, mcw 1, csb .5, eta .05
# 0.4762709 (valid 0.4810847) md 7, ss .9, mcw 1, csb .5, eta .05
# 0.476381 (valid 0.4806978) md 5, ss .9, mcw 1, csb .5, eta .05
# 0.4773583 (valid 0.4808638) md 6, ss .9, mcw 1, csb .5, eta .05
# 0.4771615 (valid 0.4806962) md 6, ss .9, mcw 3, csb .5, eta .05
# 0.4773307 (valid 0.4811267) md 6, ss .9, mcw 6, csb .5, eta .05
# 0.4762546 (valid 0.4809545) md 6, ss .9, mcw 4, csb .5, eta .05
# 0.4769558 (valid 0.4807347) md 6, ss .9, mcw 5, csb .5, eta .05
# 0.4769797 (valid 0.4807312) md 6, ss .8, mcw 4, csb .5, eta .05
# 0.4763141 (valid 0.4811592) md 6, ss .7, mcw 4, csb .5, eta .05
# 0.4764065 (valid 0.4808862) md 6, ss .6, mcw 4, csb .5, eta .05
# 0.4766367 (valid 0.4807003) md 6, ss .9, mcw 4, csb .2, eta .05
# 0.4765094 (valid 0.4804538) md 6, ss .9, mcw 4, csb .4, eta .05
# 0.4763389 (valid 0.4810005) md 6, ss .9, mcw 4, csb .6, eta .05
# 0.4763036 (valid 0.480577) md 6, ss .9, mcw 4, csb .5, eta .025 *
# best -----------------
# , md = 6
# , ss = .9
# , mcw = 4
# , csb = .5
# , eta = .025
# ----------------------


