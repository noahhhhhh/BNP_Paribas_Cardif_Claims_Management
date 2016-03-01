setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(caTools)
require(scales)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_imputed.RData")
#######################################################################################
## 1.0 remove linear dependencies #####################################################
#######################################################################################
# categorical variables v91 and v107 seems to be identical only different level names. -- Jesse Burström  
# findLinearCombos(model.matrix(target ~., dt.imputed))
# $remove
# [1]  46  48  51  52  53  54  55  56 192 193 194 195 196 197 200 609 629 637 644 654 661 665 666 670 671 678 679 681 682
# [30] 683 684 686 690 692 693 694 695 696 697
## check factor columns are the same
lapply(dt.imputed[, cols.factor, with = F], function(x) as.vector(table(x)[order(table(x))]))
# $v91toImputed
# [1]   449  6375 27035 45274 46327 49223 54031
# 
# $v107toImputed
# [1]   449  6375 27035 45274 46327 49223 54031

## remove $v107toImputed
dt.imputed[, "v107toImputed" := NULL]
cols.factor <- cols.factor[!cols.factor %in% "v107toImputed"]
## check numeric columns are the same
lapply(dt.imputed[, cols.numeric, with = F], function(x) as.vector(summary(x)))

#######################################################################################
## 2.0 basic stats about a row ########################################################
#######################################################################################
## integer
vIntMean <- rowMeans(dt.imputed[, cols.integer, with = F])
vIntMax <- apply(dt.imputed[, cols.integer, with = F], 1, max)
vIntMin <- apply(dt.imputed[, cols.integer, with = F], 1, min)
vIntSd <- apply(dt.imputed[, cols.integer, with = F], 1, sd)
cols.numeric <- c(cols.numeric, "vIntMean", "vIntMax", "vIntMin", "vIntSd")
## numeric
vNumMean <- rowMeans(dt.imputed[, cols.numeric, with = F])
vNumMax <- apply(dt.imputed[, cols.numeric, with = F], 1, max)
vNumMin <- apply(dt.imputed[, cols.numeric, with = F], 1, min)
vNumSd <- apply(dt.imputed[, cols.numeric, with = F], 1, sd)
cols.numeric <- c(cols.numeric, "vNumMean", "vNumMax", "vNumMin", "vNumSd")
## factor
# sum of ranking of individual factor out of range(dt.imputed$v..)
dt.factorRank <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.factor)
dt.factorRank <- dt.factorRank[, lapply(.SD, as.numeric)]
pre.factorRank <- preProcess(dt.factorRank
                             , method = c("range")
                             , verbose = T)
dt.factorRank.range <- predict(pre.factorRank, dt.factorRank)
vFactorRankSum <- rowSums(dt.factorRank.range)
cols.numeric <- c(cols.numeric, "vFactorRankSum")

#######################################################################################
## 3.0 encode #########################################################################
#######################################################################################
# columns need encoding
cols.needEncode <- names(ColUnique(dt.imputed[, cols.factor, with = F]))[ColUnique(dt.imputed[, cols.factor, with = F]) >= 20]
# factor encode
dt.encode.factor <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.needEncode)
setnames(dt.encode.factor, names(dt.encode.factor), paste(names(dt.encode.factor), "_factor", sep = ""))

# numeric encod
dt.encode.numeric <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.needEncode)
dt.encode.numeric <- dt.encode.numeric[, lapply(.SD, as.numeric)]
setnames(dt.encode.numeric, names(dt.encode.numeric), paste(names(dt.encode.numeric), "_numeric", sep = ""))

# remove original vars and cbind the encoded vars
dt.imputed <- dt.imputed[, names(dt.imputed)[!names(dt.imputed) %in% cols.needEncode], with = F]
cols.factor <- cols.factor[!cols.factor %in% cols.needEncode]
dt.imputed <- cbind(dt.imputed, dt.encode.factor, dt.encode.numeric)
cols.factor <- c(cols.factor, names(dt.encode.factor))
cols.numeric <- c(cols.numeric, names(dt.encode.numeric))
dim(dt.imputed)
# [1] 228714    141

#######################################################################################
## 4.0 integer 0 ######################################################################
#######################################################################################
unlist(lapply(dt.imputed[, cols.integer, with = F], function(x) sum(x == 0, na.rm = T)))
# v38    v62    v72   v129 
# 219598  41055   6735 180678 
vIntegerZero <- rowSums(dt.imputed[, cols.integer, with = F] == 0)
cols.numeric <- c(cols.numeric, "vIntegerZero")
dt.imputed[, vIntegerZero := vIntegerZero]

#######################################################################################
## save ###############################################################################
#######################################################################################
dt.featureEngineered <- dt.imputed
save(dt.featureEngineered, cols.factor, cols.numeric, cols.integer, file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_featureEngineered.RData")





