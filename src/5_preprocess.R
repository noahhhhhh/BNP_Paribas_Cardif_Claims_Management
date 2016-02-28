setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_imputed.RData")
#######################################################################################
## 1.0 impute #########################################################################
#######################################################################################