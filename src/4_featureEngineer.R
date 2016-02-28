setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_imputed.RData")
#######################################################################################
## 1.0 encode #########################################################################
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