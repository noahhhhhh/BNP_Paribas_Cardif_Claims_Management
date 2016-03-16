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
## 1.0 add basic stats about a row ####################################################
#######################################################################################
# ## integer
# vIntMean <- rowMeans(dt.imputed[, cols.integer[! cols.integer %in% cols.newFeatures], with = F])
# vIntMax <- apply(dt.imputed[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, max)
# vIntMin <- apply(dt.imputed[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, min)
# vIntSd <- apply(dt.imputed[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, sd)
# 
# ## numeric
# vNumMean <- rowMeans(dt.imputed[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F])
# vNumMax <- apply(dt.imputed[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, max)
# vNumMin <- apply(dt.imputed[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, min)
# vNumSd <- apply(dt.imputed[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, sd)

## factor
# sum of ranking of individual factor out of range(dt.imputed$v..)
# dt.factorRank <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.factor[! cols.factor %in% cols.newFeatures])
# dt.factorRank <- dt.factorRank[, lapply(.SD, as.numeric)]
# pre.factorRank <- preProcess(dt.factorRank
#                              , method = c("range")
#                              , verbose = T)
# dt.factorRank.range <- predict(pre.factorRank, dt.factorRank)
# vFactorRankSum <- rowSums(dt.factorRank.range)

#######################################################################################
## 2.0 add integer 0 ##################################################################
#######################################################################################
unlist(lapply(dt.imputed[, cols.integer, with = F], function(x) sum(x == 0, na.rm = T)))
# v38    v62    v72   v129 
# 219598  41055   6735 180678 
vIntegerZero <- rowSums(dt.imputed[, cols.integer[! cols.integer %in% cols.newFeatures], with = F] == 0)

#######################################################################################
## 2.0 correlated, duplicates, and chained ############################################
#######################################################################################
# categorical variables v91 and v107 seems to be identical only different level names. -- Jesse Burström  
# findLinearCombos(model.matrix(target ~., dt.imputed))
# $remove
# [1]  46  48  51  52  53  54  55  56 192 193 194 195 196 197 200 609 629 637 644 654 661 665 666 670 671 678 679 681 682
# [30] 683 684 686 690 692 693 694 695 696 697
## check factor columns are the same
# lapply(dt.imputed[, cols.factor, with = F], function(x) as.vector(table(x)[order(table(x))]))
# $v91toImputed
# [1]   449  6375 27035 45274 46327 49223 54031
# 
# $v107toImputed
# [1]   449  6375 27035 45274 46327 49223 54031

## remove $v107toImputed
dt.imputed[, "v107toImputed" := NULL]
cols.factor <- cols.factor[!cols.factor %in% "v107toImputed"]

## check numeric columns are the same
# lapply(dt.imputed[, cols.numeric, with = F], function(x) as.vector(summary(x)))

## merge v71 and v75
head(dt.imputed$v71)
head(dt.imputed$v75)
v71_v75 <- paste0(dt.imputed$v71, dt.imputed$v75)
head(v71_v75)
dt.imputed[, c("v71", "v75") := list(NULL, NULL)]
cols.factor <- cols.factor[!cols.factor %in% c("v71", "v75")]

## pca on v25 (v46, v54, v63, v105, v8)
md.pca.chain <- prcomp(dt.imputed[, c("v25toImputed", "v46toImputed", "v54toImputed", "v63toImputed", "v105toImputed", "v8toImputed"), with = F], scale. = T)

pc.all.chain <- md.pca.chain$x
pca.var.chain <- md.pca.chain$sdev^2
pve.chain <- pca.var.chain/sum(pca.var.chain)

plot(pve.chain[1:10] , xlab =" Principal Component ", ylab=" Proportion of
Variance Explained ", ylim=c(0,1) ,type = 'b')

plot(cumsum(pve.chain[1:10]), xlab=" Principal Component ", ylab ="Cumulative Proportion of
     Variance Explained ", ylim=c(0,1) ,type = 'b')

v8_v25_v46_v54_v63_v105_pca <- pc.all.chain[, 1]
dt.imputed[, c("v25toImputed", "v46toImputed", "v54toImputed", "v63toImputed", "v105toImputed", "v8toImputed") :=
               list(NULL, NULL, NULL, NULL, NULL, NULL)]
cols.numeric <- cols.numeric[!cols.numeric %in% c("v25toImputed", "v46toImputed", "v54toImputed", "v63toImputed", "v105toImputed", "v8toImputed")]
## chain analysis to be continued

# #######################################################################################
# ## 3.0 factor to sum of targets #######################################################
# #######################################################################################
# # columns need encoding
# cols.needEncode <- names(ColUnique(dt.imputed[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]))[ColUnique(dt.imputed[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]) > 10]
# dt.imputed <- ConvertNonNumFactorToSumOfTargets(dt.imputed, cols.needEncode)
# cols.numeric <- c(cols.numeric, paste0(cols.needEncode, "_Sum0"), paste0(cols.needEncode, "Sum1"))

#######################################################################################
## 4.0 encode #########################################################################
#######################################################################################
# # columns need encoding
# cols.needEncode <- names(ColUnique(dt.imputed[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]))[ColUnique(dt.imputed[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]) > 10]
# # factor encode
# dt.encode.factor <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.needEncode)
# setnames(dt.encode.factor, names(dt.encode.factor), paste(names(dt.encode.factor), "_factor", sep = ""))
# 
# # numeric encode
# dt.encode.numeric <- ConvertNonNumFactorToOrderedNum(dt.imputed, cols.needEncode)
# dt.encode.numeric <- dt.encode.numeric[, lapply(.SD, as.numeric)]
# setnames(dt.encode.numeric, names(dt.encode.numeric), paste(names(dt.encode.numeric), "_numeric", sep = ""))
# 
# # remove original vars and cbind the encoded vars
# dt.imputed <- dt.imputed[, names(dt.imputed)[!names(dt.imputed) %in% cols.needEncode], with = F]
# cols.factor <- cols.factor[!cols.factor %in% cols.needEncode]
# dt.imputed <- cbind(dt.imputed, dt.encode.factor, dt.encode.numeric)
# cols.factor <- c(cols.factor, names(dt.encode.factor))
# cols.numeric <- c(cols.numeric, names(dt.encode.numeric))
# dim(dt.imputed)
# [1] 228714    141

#######################################################################################
## save ###############################################################################
#######################################################################################
dt.imputed[, c(
                "v71_v75", "v8_v25_v46_v54_v63_v105_pca"
               # , "vIntMean", "vIntMax", "vIntMin", "vIntSd"
               # , "vNumMean", "vNumMax", "vNumMin", "vNumSd"
               # , "vFactorRankSum"
               , "vIntegerZero"
               ) := list(
                                         v71_v75, v8_v25_v46_v54_v63_v105_pca
                                         # , vIntMean, vIntMax, vIntMin, vIntSd
                                         # , vNumMean, vNumMax, vNumMin, vNumSd
                                         # , vFactorRankSum
                                         , vIntegerZero)]
cols.numeric <- c(cols.numeric, "v8_v25_v46_v54_v63_v105_pca")
cols.factor <- c(cols.factor, "v71_v75")
# cols.numeric <- c(cols.numeric, "vIntMean", "vIntMax", "vIntMin", "vIntSd")
# cols.numeric <- c(cols.numeric, "vNumMean", "vNumMax", "vNumMin", "vNumSd")
# cols.numeric <- c(cols.numeric, "vFactorRankSum")
cols.numeric <- c(cols.numeric, "vIntegerZero")
cols.newFeatures <- c(cols.newFeatures
                      , "v71_v75", "v8_v25_v46_v54_v63_v105_pca"
                      # , "vIntMean", "vIntMax", "vIntMin", "vIntSd"
                      # , "vNumMean", "vNumMax", "vNumMin", "vNumSd"
                      # , "vFactorRankSum"
                      , "vIntegerZero")
dt.featureEngineered <- dt.imputed

cols.basicStats <- c(
                    # "vIntMean", "vIntMax", "vIntMin", "vIntSd"
                     # , "vNumMean", "vNumMax", "vNumMin", "vNumSd"
                     # "vFactorRankSum"
    )
cols.analysis <- c("v71_v75", "v8_v25_v46_v54_v63_v105_pca")
cols.zero <- c("vIntegerZero")

save(dt.featureEngineered, cols.factor, cols.numeric, cols.integer
     # , cols.needEncode
     , cols.newFeatures
     , cols.basicStats, cols.analysis, cols.zero
     , file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_featureEngineered.RData")





