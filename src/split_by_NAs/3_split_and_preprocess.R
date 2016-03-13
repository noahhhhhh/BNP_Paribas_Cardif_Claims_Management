setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_featureEngineered.RData")
#######################################################################################
## 1.0 classify #######################################################################
#######################################################################################
MyClassify <- function(dt, cols.factor){
    dt.classFactor <- dt[, cols.factor, with = F][, lapply(.SD, as.factor)]
    dt <- data.table(dt[, !cols.factor, with = F], dt.classFactor)
    return(dt)
}
dt.group.A <- MyClassify(dt.group.A, cols.A.factor)
dt.group.B <- MyClassify(dt.group.B, cols.B.factor)
dt.group.C <- MyClassify(dt.group.C, cols.C.factor)

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
save(dt.group.A, dt.group.B, dt.group.C
     , cols.A.factor, cols.A.numeric, cols.A.integer, cols.split.A.newFeatures
     , cols.B.factor, cols.B.numeric, cols.B.integer, cols.split.B.newFeatures
     , cols.C.factor, cols.C.numeric, cols.C.integer, cols.split.C.newFeatures
     , file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_preprocessed.RData")



