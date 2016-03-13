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

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_explored.RData")
#######################################################################################
## 1.0 groups by NO. of NAs ###########################################################
#######################################################################################
## below is majorily based on JohnM's findings
stats.NAs <- ColNAs(dt.explored)
stats.NAs[order(stats.NAs)]
## groups
cols.zero.NAs <- names(stats.NAs[stats.NAs == 0])
cols.zero.NAs <- cols.zero.NAs[!cols.zero.NAs %in% c("ID", "target")]
cols.minor.NAs <- names(stats.NAs[stats.NAs > 0 & stats.NAs <= 13800])
cols.many.NAs <- names(stats.NAs[stats.NAs > 13800 & stats.NAs <= 102678])
cols.random.NAs <- names(stats.NAs[stats.NAs > 102678])

## sub groups
cols.many.NAs.A <- names(stats.NAs[stats.NAs > 13800 & stats.NAs < 99635])
cols.many.NAs.B <- names(stats.NAs[stats.NAs >= 99635 & stats.NAs <= 102678])

## distinct groups
cols.A <- unique(c(cols.zero.NAs, cols.minor.NAs))
cols.B <- cols.many.NAs.A
cols.C <- cols.many.NAs.B
cols.D <- cols.random.NAs
length(cols.A) + 2 + length(cols.B) + length(cols.C) + length(cols.D)
# 133
dim(dt.explored)[2]
# 133

## final groups
cols.group.A <- c(cols.A, cols.D)
cols.group.B <- c(cols.A, cols.D, cols.B)
cols.group.C <- c(cols.A, cols.D, cols.C)

#######################################################################################
## 2.0 split into 3 groups ############################################################
#######################################################################################
dt.group.A <- dt.explored[, c("ID", "target", cols.group.A), with = F]
dt.group.B <- dt.explored[, c("ID", "target", cols.group.B), with = F]
dt.group.C <- dt.explored[, c("ID", "target", cols.group.C), with = F]
dim(dt.group.A)[2] + dim(dt.group.B)[2] + dim(dt.group.C)[2] - (2 + length(cols.A) + length(cols.D)) * 2
# 133
dim(dt.explored)[2]
# 133

#######################################################################################
## 3.0 deal with NA rows ##############################################################
#######################################################################################
## add some features about NAs
vNa_minor <- rowSums(is.na(dt.group.A[, !cols.D, with = F]))
vNa_random <- rowSums(is.na(dt.group.A[, cols.D, with = F]))
vNa_A <- vNa_minor + vNa_random

## dt.group.A - impute NAs with -1
dt.group.A[, c("vNa_minor", "vNa_random", "vNa_A") := list(vNa_minor, vNa_random, vNa_A)]
cols.NAs.group.A <- names(ColNAs(dt.group.A, output = "nonZero"))
ls.imputed.A <- MyImpute(dt.group.A, cols.NAs.group.A, impute_type = "-1")
dt.group.A <- data.table(dt.group.A[, !cols.NAs.group.A, with = F], ls.imputed.A[[1]])

## dt.group.B - remove common NA rows and impute the rest NAs with -1
dt.group.B <- data.table(dt.group.A, dt.group.B[, !c("ID", "target", cols.group.A), with = F])
dt.group.B.temp <- dt.group.B[, cols.B, with = F]
dim(dt.group.B.temp)[2]
# 19
ind.complete.cases <- rowSums(is.na(dt.group.B.temp)) != dim(dt.group.B.temp)[2]
dt.group.B <- dt.group.B[ind.complete.cases]
# add some features about NAs in dt.group.B
vNa_B_only <- rowSums(is.na(dt.group.B))
vNa_B_minor <- vNa_B_only + vNa_minor[ind.complete.cases]
vNa_B_random <- vNa_B_only + vNa_random[ind.complete.cases]
vNa_B_A <- vNa_B_only + vNa_A[ind.complete.cases]
# impute
dt.group.B[, c("vNa_minor", "vNa_random", "vNa_A") := list(vNa_minor, vNa_random, vNa_A)]
dt.group.B[, c("vNa_B_only", "vNa_B_minor", "vNa_B_random", "vNa_B_A") := list(vNa_B_only, vNa_B_minor, vNa_B_random, vNa_B_A)]
cols.NAs.group.B <- names(ColNAs(dt.group.B, output = "nonZero"))
ls.imputed.B <- MyImpute(dt.group.B, cols.NAs.group.B, impute_type = "-1")
dt.group.B <- data.table(dt.group.B[, !cols.NAs.group.B, with = F], ls.imputed.B[[1]])

## dt.group.C - remove common NA rows and impute the rest NAs with -1
dt.group.C <- data.table(dt.group.A, dt.group.C[, !c("ID", "target", cols.group.A), with = F])
dt.group.C.temp <- dt.group.C[, cols.C, with = F]
dim(dt.group.C.temp)[2]
# 81
ind.complete.cases <- rowSums(is.na(dt.group.C.temp)) != dim(dt.group.C.temp)[2]
dt.group.C <- dt.group.C[ind.complete.cases]
# add some features about NAs in dt.group.C
vNa_C_only <- rowSums(is.na(dt.group.C))
vNa_C_minor <- vNa_C_only + vNa_minor[ind.complete.cases]
vNa_C_random <- vNa_C_only + vNa_random[ind.complete.cases]
vNa_C_A <- vNa_C_only + vNa_A[ind.complete.cases]
# impute
dt.group.C[, c("vNa_minor", "vNa_random", "vNa_A") := list(vNa_minor, vNa_random, vNa_A)]
dt.group.C[, c("vNa_C_only", "vNa_C_minor", "vNa_C_random", "vNa_C_A") := list(vNa_C_only, vNa_C_minor, vNa_C_random, vNa_C_A)]
cols.NAs.group.C <- names(ColNAs(dt.group.C, output = "nonZero"))
ls.imputed.C <- MyImpute(dt.group.C, cols.NAs.group.C, impute_type = "-1")
dt.group.C <- data.table(dt.group.C[, !cols.NAs.group.C, with = F], ls.imputed.C[[1]])

#######################################################################################
## save ###############################################################################
#######################################################################################
cols.split.A.newFeatures <- c("vNa_minor", "vNa_random", "vNa_A")
cols.split.B.newFeatures <- c(c("vNa_minor", "vNa_random", "vNa_A")
                              , c("vNa_B_only", "vNa_B_minor", "vNa_B_random", "vNa_B_A"))
cols.split.C.newFeatures <- c(c("vNa_minor", "vNa_random", "vNa_A")
                              , c("vNa_C_only", "vNa_C_minor", "vNa_C_random", "vNa_C_A"))

save(dt.group.A, dt.group.B, dt.group.C
     , cols.split.A.newFeatures, cols.split.B.newFeatures, cols.split.C.newFeatures
     , file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_impute.RData")






