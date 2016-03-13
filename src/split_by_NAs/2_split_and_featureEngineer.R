setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(caTools)
require(scales)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_impute.RData")
#######################################################################################
## 0.0 column names by class ##########################################################
#######################################################################################
## dt.group.A
class <- unlist(lapply(dt.group.A, class))
table((class)[!names(class) %in% c("ID", "target")])
# character   integer   numeric 
# 19         4        11 
cols.A.factor <- names(class)[class == "character"]
cols.A.numeric <- names(class)[class == "numeric"][names(class)[class == "numeric"] != "target"]
cols.A.integer <- names(class)[class == "integer"][names(class)[class == "integer"] != "ID"]

## dt.group.B
class <- unlist(lapply(dt.group.B, class))
table((class)[!names(class) %in% c("ID", "target")])
# character   integer   numeric 
# 19         4        34
cols.B.factor <- names(class)[class == "character"]
cols.B.numeric <- names(class)[class == "numeric"][names(class)[class == "numeric"] != "target"]
cols.B.integer <- names(class)[class == "integer"][names(class)[class == "integer"] != "ID"]

## dt.group.C
class <- unlist(lapply(dt.group.C, class))
table((class)[!names(class) %in% c("ID", "target")])
# character   integer   numeric 
# 19         4        96
cols.C.factor <- names(class)[class == "character"]
cols.C.numeric <- names(class)[class == "numeric"][names(class)[class == "numeric"] != "target"]
cols.C.integer <- names(class)[class == "integer"][names(class)[class == "integer"] != "ID"]

#######################################################################################
## 1.0 add basic stats about a row ####################################################
#######################################################################################
MyEngginer_Stats <- function(dt, cols.integer, cols.numeric, cols.factor, cols.newFeatures){
    ## integer
    cat("integer ...")
    vIntMean <- rowMeans(dt[, cols.integer[! cols.integer %in% cols.newFeatures], with = F])
    vIntMax <- apply(dt[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, max)
    vIntMin <- apply(dt[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, min)
    vIntSd <- apply(dt[, cols.integer[! cols.integer %in% cols.newFeatures], with = F], 1, sd)
    
    ## numeric
    cat("numeric ...")
    vNumMean <- rowMeans(dt[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F])
    vNumMax <- apply(dt[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, max)
    vNumMin <- apply(dt[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, min)
    vNumSd <- apply(dt[, cols.numeric[! cols.numeric %in% cols.newFeatures], with = F], 1, sd)
    
    ## factor
    cat("factor ...")
    # sum of ranking of individual factor out of range(dt$v..)
    dt.factorRank <- ConvertNonNumFactorToOrderedNum(dt, cols.factor[! cols.factor %in% cols.newFeatures])
    dt.factorRank <- dt.factorRank[, lapply(.SD, as.numeric)]
    pre.factorRank <- preProcess(dt.factorRank
                                 , method = c("range")
                                 , verbose = F)
    dt.factorRank.range <- predict(pre.factorRank, dt.factorRank)
    vFactorRankSum <- rowSums(dt.factorRank.range)
    
    dt[, c("vIntMean", "vIntMax", "vIntMin", "vIntSd"
           , "vNumMean", "vNumMax", "vNumMin", "vNumSd"
           , "vFactorRankSum") := list(vIntMean, vIntMax, vIntMin, vIntSd
                                      , vNumMean, vNumMax, vNumMin, vNumSd
                                      , vFactorRankSum)]
    cols.new <- c("vIntMean", "vIntMax", "vIntMin", "vIntSd"
                  , "vNumMean", "vNumMax", "vNumMin", "vNumSd"
                  , "vFactorRankSum")
    return(list(dt, cols.new))
}

ls.stats.A <- MyEngginer_Stats(dt.group.A, cols.A.integer, cols.A.numeric, cols.A.factor, cols.split.A.newFeatures)
ls.stats.B <- MyEngginer_Stats(dt.group.B, cols.B.integer, cols.B.numeric, cols.B.factor, cols.split.B.newFeatures)
ls.stats.C <- MyEngginer_Stats(dt.group.C, cols.C.integer, cols.C.numeric, cols.C.factor, cols.split.C.newFeatures)

dt.group.A <- ls.stats.A[[1]]
cols.split.A.newFeatures <- c(cols.split.A.newFeatures, ls.stats.A[[2]])
dt.group.B <- ls.stats.B[[1]]
cols.split.B.newFeatures <- c(cols.split.B.newFeatures, ls.stats.B[[2]])
dt.group.C <- ls.stats.C[[1]]
cols.split.C.newFeatures <- c(cols.split.C.newFeatures, ls.stats.C[[2]])
#######################################################################################
## 2.0 add integer 0 ##################################################################
#######################################################################################
MyEngginer_Int0 <- function(dt, cols.integer, cols.newFeatures){
    vIntegerZero <- rowSums(dt[, cols.integer[! cols.integer %in% cols.newFeatures], with = F] == 0)
    dt[, "vIntegerZero" := vIntegerZero]
    cols.new <- "vIntegerZero"
    return(list(dt, cols.new))
}
ls.int0.A <- MyEngginer_Int0(dt.group.A, cols.A.integer, cols.split.A.newFeatures)
ls.int0.B <- MyEngginer_Int0(dt.group.B, cols.B.integer, cols.split.B.newFeatures)
ls.int0.C <- MyEngginer_Int0(dt.group.C, cols.C.integer, cols.split.C.newFeatures)

dt.group.A <- ls.int0.A[[1]]
cols.split.A.newFeatures <- c(cols.split.A.newFeatures, ls.int0.A[[2]])
dt.group.B <- ls.int0.B[[1]]
cols.split.B.newFeatures <- c(cols.split.B.newFeatures, ls.int0.B[[2]])
dt.group.C <- ls.int0.C[[1]]
cols.split.C.newFeatures <- c(cols.split.C.newFeatures, ls.int0.C[[2]])
#######################################################################################
## 2.0 correlated, duplicates, and chained ############################################
#######################################################################################
## remove $v107toImputed
dt.group.A[, "v107toImputed" := NULL]
cols.A.factor <- cols.A.factor[!cols.A.factor %in% "v107toImputed"]

#######################################################################################
## 3.0 encode #########################################################################
#######################################################################################
MyEngginer_Encode <- function(dt, cols.factor, cols.numeric, cols.newFeatures, limit = 20){
    # columns need encoding
    cols.needEncode <- names(ColUnique(dt[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]))[ColUnique(dt[, cols.factor[! cols.factor %in% cols.newFeatures], with = F]) >= limit]
    # factor encode
    dt.encode.factor <- ConvertNonNumFactorToOrderedNum(dt, cols.needEncode)
    setnames(dt.encode.factor, names(dt.encode.factor), paste(names(dt.encode.factor), "_factor", sep = ""))
    
    # numeric encode
    dt.encode.numeric <- ConvertNonNumFactorToOrderedNum(dt, cols.needEncode)
    dt.encode.numeric <- dt.encode.numeric[, lapply(.SD, as.numeric)]
    setnames(dt.encode.numeric, names(dt.encode.numeric), paste(names(dt.encode.numeric), "_numeric", sep = ""))
    
    # remove original vars and cbind the encoded vars
    dt <- dt[, names(dt)[!names(dt) %in% cols.needEncode], with = F]
    cols.factor <- cols.factor[!cols.factor %in% cols.needEncode]
    dt <- cbind(dt, dt.encode.factor, dt.encode.numeric)
    cols.factor <- c(cols.factor, names(dt.encode.factor))
    cols.numeric <- c(cols.numeric, names(dt.encode.numeric))
    dim(dt)
    # [1] 228714    141
    return(list(dt, cols.factor, cols.numeric))
}

ls.A.encode <- MyEngginer_Encode(dt.group.A, cols.A.factor, cols.A.numeric, cols.split.A.newFeatures, 20)
ls.B.encode <- MyEngginer_Encode(dt.group.B, cols.B.factor, cols.B.numeric, cols.split.B.newFeatures, 20)
ls.C.encode <- MyEngginer_Encode(dt.group.C, cols.C.factor, cols.C.numeric, cols.split.C.newFeatures, 20)

dt.group.A <- ls.A.encode[[1]]
cols.A.factor <- ls.A.encode[[2]]
cols.A.numeric <- ls.A.encode[[3]]
dt.group.B <- ls.B.encode[[1]]
cols.B.factor <- ls.B.encode[[2]]
cols.B.numeric <- ls.B.encode[[3]]
dt.group.C <- ls.C.encode[[1]]
cols.C.factor <- ls.C.encode[[2]]
cols.C.numeric <- ls.C.encode[[3]]

#######################################################################################
## save ###############################################################################
#######################################################################################
save(dt.group.A, dt.group.B, dt.group.C
     , cols.A.factor, cols.A.numeric, cols.A.integer, cols.split.A.newFeatures
     , cols.B.factor, cols.B.numeric, cols.B.integer, cols.split.B.newFeatures
     , cols.C.factor, cols.C.numeric, cols.C.integer, cols.split.C.newFeatures
     , file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_split_and_featureEngineered.RData")





