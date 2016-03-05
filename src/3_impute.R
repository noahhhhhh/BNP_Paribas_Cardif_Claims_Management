setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_explored.RData")
#######################################################################################
## 1.0 impute #########################################################################
#######################################################################################
cols.na <- ColNAs(dt.explored, output = "nonZero")
cols.na <- names(cols.na)

## factor NA columns
cols.na.factor <- intersect(cols.factor, cols.na)
# [1] "v3"   "v22"  "v30"  "v31"  "v52"  "v56"  "v91"  "v107" "v112" "v113" "v125"
ColNAs(dt.explored[, cols.na.factor, with = F], method = "mean")
# v3  v22  v30  v31  v52  v56  v91 v107 v112 v113 v125 
# 0.03 0.00 0.53 0.03 0.00 0.06 0.00 0.00 0.00 0.48 0.00

## integer NA columns
cols.na.integer <- intersect(cols.integer, cols.na)
# character(0)

## numeric NA columns
cols.na.numeric <- intersect(cols.numeric, cols.na)
# [1] "v1"   "v2"   "v4"   "v5"   "v6"   "v7"   "v8"   "v9"   "v10"  "v11"  "v12"  "v13"  "v14"  "v15"  "v16"  "v17" 
# [17] "v18"  "v19"  "v20"  "v21"  "v23"  "v25"  "v26"  "v27"  "v28"  "v29"  "v32"  "v33"  "v34"  "v35"  "v36"  "v37" 
# [33] "v39"  "v40"  "v41"  "v42"  "v43"  "v44"  "v45"  "v46"  "v48"  "v49"  "v50"  "v51"  "v53"  "v54"  "v55"  "v57" 
# [49] "v58"  "v59"  "v60"  "v61"  "v63"  "v64"  "v65"  "v67"  "v68"  "v69"  "v70"  "v73"  "v76"  "v77"  "v78"  "v80" 
# [65] "v81"  "v82"  "v83"  "v84"  "v85"  "v86"  "v87"  "v88"  "v89"  "v90"  "v92"  "v93"  "v94"  "v95"  "v96"  "v97" 
# [81] "v98"  "v99"  "v100" "v101" "v102" "v103" "v104" "v105" "v106" "v108" "v109" "v111" "v114" "v115" "v116" "v117"
# [97] "v118" "v119" "v120" "v121" "v122" "v123" "v124" "v126" "v127" "v128" "v130" "v131"
ColNAs(dt.explored[, cols.na.numeric, with = F], method = "mean")
# v1   v2   v4   v5   v6   v7   v8   v9  v10  v11  v12  v13  v14  v15  v16  v17  v18  v19  v20  v21  v23  v25  v26  v27 
# 0.44 0.44 0.44 0.43 0.44 0.44 0.43 0.44 0.00 0.44 0.00 0.44 0.00 0.44 0.44 0.44 0.44 0.44 0.44 0.01 0.44 0.43 0.44 0.44 
# v28  v29  v32  v33  v34  v35  v36  v37  v39  v40  v41  v42  v43  v44  v45  v46  v48  v49  v50  v51  v53  v54  v55  v57 
# 0.44 0.44 0.44 0.44 0.00 0.44 0.43 0.44 0.44 0.00 0.44 0.44 0.44 0.44 0.44 0.43 0.44 0.44 0.00 0.44 0.44 0.43 0.44 0.44 
# v58  v59  v60  v61  v63  v64  v65  v67  v68  v69  v70  v73  v76  v77  v78  v80  v81  v82  v83  v84  v85  v86  v87  v88 
# 0.44 0.44 0.44 0.44 0.43 0.44 0.44 0.44 0.44 0.44 0.43 0.44 0.44 0.44 0.44 0.44 0.43 0.43 0.44 0.44 0.44 0.44 0.43 0.44 
# v89  v90  v92  v93  v94  v95  v96  v97  v98  v99 v100 v101 v102 v103 v104 v105 v106 v108 v109 v111 v114 v115 v116 v117 
# 0.43 0.44 0.44 0.44 0.44 0.44 0.44 0.44 0.43 0.44 0.44 0.44 0.45 0.44 0.44 0.43 0.44 0.43 0.43 0.44 0.00 0.44 0.44 0.43 
# v118 v119 v120 v121 v122 v123 v124 v126 v127 v128 v130 v131 
# 0.44 0.44 0.44 0.44 0.44 0.44 0.43 0.44 0.44 0.43 0.44 0.44 

## create features for NAs
vNaFactor <- rowSums(is.na(dt.explored[, cols.na.factor, with = F]))
vNaNumeric <- rowSums(is.na(dt.explored[, cols.na.numeric, with = F]))
vNa <- rowSums(is.na(dt.explored))
cols.numeric <- c(cols.numeric, c("vNaFactor", "vNaNumeric", "vNa"))
cols.newFeatures <- c("vNaFactor", "vNaNumeric", "vNa")

## impute with median
ls.imputed <- MyImpute(dt.explored, c(cols.na.numeric, cols.na.factor), impute_type = "-1")

## remove original vars and cbind the imputed vars
dt.explored <- dt.explored[, names(dt.explored)[!names(dt.explored) %in% c(cols.na.numeric, cols.na.factor)] , with = F]
cols.numeric <- cols.numeric[!cols.numeric %in% cols.na.numeric]
cols.factor <- cols.factor[!cols.factor %in% cols.na.factor]

dt.explored <- cbind(dt.explored, ls.imputed[[1]])
cols.numeric <- c(cols.numeric, paste(cols.na.numeric, "toImputed", sep = ""))
cols.factor <- c(cols.factor, paste(cols.na.factor, "toImputed", sep = ""))

## cbind the created features about NAs
dt.explored <- cbind(dt.explored, vNaFactor, vNaNumeric, vNa)
dim(dt.explored)

#######################################################################################
## 2.0 save ###########################################################################
#######################################################################################
dt.imputed <- dt.explored
save(dt.imputed, cols.factor, cols.numeric, cols.integer, cols.newFeatures, file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_imputed.RData")

