setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
#######################################################################################
## 1.0 read ###########################################################################
#######################################################################################
dt.train.raw <- fread("../data/BNP_Paribas_Cardif_Claims_Management/train.csv", stringsAsFactors = T, na.strings = c("NA", ""))
dt.test.raw <- fread("../data/BNP_Paribas_Cardif_Claims_Management/test.csv", stringsAsFactors = T, na.strings = c("NA", ""))
dim(dt.train.raw); dim(dt.test.raw)
# [1] 114321    133
# [1] 114393    132

## check the balance of target of dt.train.raw
table(dt.train.raw$target)
# 0     1 
# 27300 87021

#######################################################################################
## 2.0 combine ########################################################################
#######################################################################################
## set -1 to target to dt.test.raw
dt.test.raw[, target := -1]
dim(dt.train.raw); dim(dt.test.raw)
# [1] 114321    133
# [1] 114393    133

## rearrange the column names of dt.test.raw
dt.test.raw <- dt.test.raw[, names(dt.train.raw), with = F]

## check if the column names are identical
identical(names(dt.train.raw), names(dt.test.raw))
# [1] TRUE

## combine
dt.all <- rbind(dt.train.raw, dt.test.raw)
dim(dt.all)
# [1] 228714    133

## check the number of dt.test.raw
dim(dt.all[dt.all$target == -1])
# [1] 114393    133
dim(dt.test.raw)
# [1] 114393    133

#######################################################################################
## 3.0 save ###########################################################################
#######################################################################################
save(dt.all, file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_all.RData")
