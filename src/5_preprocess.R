setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_featureEngineered.RData")
#######################################################################################
## 1.0 classify #######################################################################
#######################################################################################
dt.classFactor <- dt.featureEngineered[, cols.factor, with = F][, lapply(.SD, as.factor)]
dt.featureEngineered <- data.table(dt.featureEngineered[, !cols.factor, with = F], dt.classFactor)

#######################################################################################
## 2.0 logify #########################################################################
#######################################################################################
# Multiple histograms
# par(mfrow=c(3, 3))
# colnames <- names(dt.featureEngineered[, cols.numeric, with = F])
# for (i in 1:length(cols.numeric)[1:9]) {
#     hist(dt.featureEngineered[, cols.numeric, with = F][[i]], main=colnames[i], probability=TRUE, col="gray", border="white")
# }
# par(mfrow = c(1, 1))
# 
# dt.featureEngineered[target != -1]

#######################################################################################
## save ###############################################################################
#######################################################################################
dt.preprocessed <- dt.featureEngineered
save(dt.preprocessed, cols.factor, cols.numeric, cols.integer, cols.newFeatures
     , cols.basicStats, cols.analysis, cols.zero
     , file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_preprocessed.RData")



