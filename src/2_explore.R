setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/BNP_Paribas_Cardif_Claims_Management/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_all.RData")
#######################################################################################
## 1.0 explore ########################################################################
#######################################################################################
## check NAs
ColNAs(dt.all, method = "mean")
# ID target     v1     v2     v3     v4     v5     v6     v7     v8     v9    v10    v11    v12    v13    v14    v15 
# 0.00   0.00   0.44   0.44   0.00   0.44   0.43   0.44   0.44   0.43   0.44   0.00   0.44   0.00   0.44   0.00   0.44 
# v16    v17    v18    v19    v20    v21    v22    v23    v24    v25    v26    v27    v28    v29    v30    v31    v32 
# 0.44   0.44   0.44   0.44   0.44   0.01   0.00   0.44   0.00   0.43   0.44   0.44   0.44   0.44   0.00   0.00   0.44 
# v33    v34    v35    v36    v37    v38    v39    v40    v41    v42    v43    v44    v45    v46    v47    v48    v49 
# 0.44   0.00   0.44   0.43   0.44   0.00   0.44   0.00   0.44   0.44   0.44   0.44   0.44   0.43   0.00   0.44   0.44 
# v50    v51    v52    v53    v54    v55    v56    v57    v58    v59    v60    v61    v62    v63    v64    v65    v66 
# 0.00   0.44   0.00   0.44   0.43   0.44   0.00   0.44   0.44   0.44   0.44   0.44   0.00   0.43   0.44   0.44   0.00 
# v67    v68    v69    v70    v71    v72    v73    v74    v75    v76    v77    v78    v79    v80    v81    v82    v83 
# 0.44   0.44   0.44   0.43   0.00   0.00   0.44   0.00   0.00   0.44   0.44   0.44   0.00   0.44   0.43   0.43   0.44 
# v84    v85    v86    v87    v88    v89    v90    v91    v92    v93    v94    v95    v96    v97    v98    v99   v100 
# 0.44   0.44   0.44   0.43   0.44   0.43   0.44   0.00   0.44   0.44   0.44   0.44   0.44   0.44   0.43   0.44   0.44 
# v101   v102   v103   v104   v105   v106   v107   v108   v109   v110   v111   v112   v113   v114   v115   v116   v117 
# 0.44   0.45   0.44   0.44   0.43   0.44   0.00   0.43   0.43   0.00   0.44   0.00   0.00   0.00   0.44   0.44   0.43 
# v118   v119   v120   v121   v122   v123   v124   v125   v126   v127   v128   v129   v130   v131 
# 0.44   0.44   0.44   0.44   0.44   0.44   0.43   0.00   0.44   0.44   0.43   0.00   0.44   0.44 

## check class
unlist(lapply(dt.all, class))
# ID    target        v1        v2        v3        v4        v5        v6        v7        v8        v9       v10 
# "integer" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" 
# v11       v12       v13       v14       v15       v16       v17       v18       v19       v20       v21       v22 
# "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric"  "factor" 
# v23       v24       v25       v26       v27       v28       v29       v30       v31       v32       v33       v34 
# "numeric"  "factor" "numeric" "numeric" "numeric" "numeric" "numeric"  "factor"  "factor" "numeric" "numeric" "numeric" 
# v35       v36       v37       v38       v39       v40       v41       v42       v43       v44       v45       v46 
# "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" 
# v47       v48       v49       v50       v51       v52       v53       v54       v55       v56       v57       v58 
# "factor" "numeric" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" 
# v59       v60       v61       v62       v63       v64       v65       v66       v67       v68       v69       v70 
# "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" "numeric" 
# v71       v72       v73       v74       v75       v76       v77       v78       v79       v80       v81       v82 
# "factor" "integer" "numeric"  "factor"  "factor" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" 
# v83       v84       v85       v86       v87       v88       v89       v90       v91       v92       v93       v94 
# "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" 
# v95       v96       v97       v98       v99      v100      v101      v102      v103      v104      v105      v106 
# "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" 
# v107      v108      v109      v110      v111      v112      v113      v114      v115      v116      v117      v118 
# "factor" "numeric" "numeric"  "factor" "numeric"  "factor"  "factor" "numeric" "numeric" "numeric" "numeric" "numeric" 
# v119      v120      v121      v122      v123      v124      v125      v126      v127      v128      v129      v130 
# "numeric" "numeric" "numeric" "numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" "integer" "numeric" 
# v131 
# "numeric" 

## summarise column class
class <- unlist(lapply(dt.all, class))
table((class)[!names(class) %in% c("ID", "target")])
# factor integer numeric 
# 19       4     108 

## save the colmmns by class
cols.factor <- names(class)[class == "factor"]
cols.numeric <- names(class)[class == "numeric"][names(class)[class == "numeric"] != "target"]
cols.integer <- names(class)[class == "integer"][names(class)[class == "integer"] != "ID"]

## check unique values
ColUnique(dt.all[, cols.factor, with = F])
# v3   v22   v24   v30   v31   v47   v52   v56   v66   v71   v74   v75   v79   v91  v107  v110  v112  v113  v125 
# 4 23420     5     8     4    10    13   131     3    12     3     4    18     8     8     3    23    38    91 

# check tabular summary of all factor columns
lapply(dt.all[, cols.factor, with = F], table)

#######################################################################################
## 2.0 save ###########################################################################
#######################################################################################
dt.explored <- dt.all
save(dt.explored, cols.factor, cols.integer, cols.numeric, file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_explored.RData")





