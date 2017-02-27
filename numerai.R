
cat("Loading libraries...\n");
library(xgboost)
library(data.table)
library(Matrix)
library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(treemap)
library(Ckmeans.1d.dp)
library(party)
library(rpart)
library(randomForest)
library(rpart.plot)
library(Metrics)
library(arules)
library(arulesViz)
library(moments)
library(tsne)


# read in Training data
cat("Loading data...\n");
numerai_training_data <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/numerai_training_data10_26.csv")
eigen <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/eigenvectors.csv")

# # Create Splits for Stacking
# require(caTools)
# set.seed(1969) 
# sample = sample.split(data$anycolumn, SplitRatio = .75)
# train = subset(data, sample == TRUE)
# test = subset(data, sample == FALSE)

# Select which Source File
nd<-data.table(numerai_training_data)

# Retain Target Prediction
rm(target)
target<-nd$target

# Add Features
nd$r1_8<-nd$feature1/nd$feature8
nd$r8_13<-nd$feature8/nd$feature13
nd$r5_6<-nd$feature5/nd$feature6
nd$r12_17<-nd$feature12/nd$feature17
#nd$r12_18<-nd$feature12/nd$feature18

# nd$r1_12<-nd$feature1/nd$feature12
# nd$r1_18<-nd$feature1/nd$feature18
# nd$r2_17<-nd$feature2/nd$feature17
# nd$r2_6<-nd$feature2/nd$feature6
# nd$r2_7<-nd$feature2/nd$feature7
# nd$r3_6<-nd$feature3/nd$feature6
# nd$r3_14<-nd$feature3/nd$feature14
# nd$r3_17<-nd$feature3/nd$feature17
# nd$r4_9<-nd$feature4/nd$feature9
# nd$r4_10<-nd$feature4/nd$feature10
# nd$r5_8<-nd$feature5/nd$feature8
# nd$r6_14<-nd$feature6/nd$feature14
# nd$r6_17<-nd$feature6/nd$feature17
# nd$r7_13<-nd$feature7/nd$feature13
# nd$r7_17<-nd$feature7/nd$feature17
# nd$r8_15<-nd$feature8/nd$feature15


# #Eigenvectors from PCA
# rm(pca)
# pca<-nd
# pca$v1<-eigen$V1[1]*pca$feature1+eigen$V1[2]*pca$feature2+eigen$V1[3]*pca$feature3+eigen$V1[4]*pca$feature4+eigen$V1[5]*pca$feature5+eigen$V1[6]*pca$feature6+eigen$V1[7]*pca$feature7+eigen$V1[8]*pca$feature8+eigen$V1[9]*pca$feature9+eigen$V1[10]*pca$feature10+eigen$V1[11]*pca$feature11+eigen$V1[12]*pca$feature12+eigen$V1[13]*pca$feature13+eigen$V1[14]*pca$feature14+eigen$V1[15]*pca$feature15+eigen$V1[16]*pca$feature16+eigen$V1[17]*pca$feature17+eigen$V1[18]*pca$feature18+eigen$V1[19]*pca$feature19+eigen$V1[20]*pca$feature20+eigen$V1[21]*pca$feature21;
# pca$v2<-eigen$V2[1]*pca$feature1+eigen$V2[2]*pca$feature2+eigen$V2[3]*pca$feature3+eigen$V2[4]*pca$feature4+eigen$V2[5]*pca$feature5+eigen$V2[6]*pca$feature6+eigen$V2[7]*pca$feature7+eigen$V2[8]*pca$feature8+eigen$V2[9]*pca$feature9+eigen$V2[10]*pca$feature10+eigen$V2[11]*pca$feature11+eigen$V2[12]*pca$feature12+eigen$V2[13]*pca$feature13+eigen$V2[14]*pca$feature14+eigen$V2[15]*pca$feature15+eigen$V2[16]*pca$feature16+eigen$V2[17]*pca$feature17+eigen$V2[18]*pca$feature18+eigen$V2[19]*pca$feature19+eigen$V2[20]*pca$feature20+eigen$V2[21]*pca$feature21;
# pca$v3<-eigen$V3[1]*pca$feature1+eigen$V3[2]*pca$feature2+eigen$V3[3]*pca$feature3+eigen$V3[4]*pca$feature4+eigen$V3[5]*pca$feature5+eigen$V3[6]*pca$feature6+eigen$V3[7]*pca$feature7+eigen$V3[8]*pca$feature8+eigen$V3[9]*pca$feature9+eigen$V3[10]*pca$feature10+eigen$V3[11]*pca$feature11+eigen$V3[12]*pca$feature12+eigen$V3[13]*pca$feature13+eigen$V3[14]*pca$feature14+eigen$V3[15]*pca$feature15+eigen$V3[16]*pca$feature16+eigen$V3[17]*pca$feature17+eigen$V3[18]*pca$feature18+eigen$V3[19]*pca$feature19+eigen$V3[20]*pca$feature20+eigen$V3[21]*pca$feature21;
# pca$v4<-eigen$V4[1]*pca$feature1+eigen$V4[2]*pca$feature2+eigen$V4[3]*pca$feature3+eigen$V4[4]*pca$feature4+eigen$V4[5]*pca$feature5+eigen$V4[6]*pca$feature6+eigen$V4[7]*pca$feature7+eigen$V4[8]*pca$feature8+eigen$V4[9]*pca$feature9+eigen$V4[10]*pca$feature10+eigen$V4[11]*pca$feature11+eigen$V4[12]*pca$feature12+eigen$V4[13]*pca$feature13+eigen$V4[14]*pca$feature14+eigen$V4[15]*pca$feature15+eigen$V4[16]*pca$feature16+eigen$V4[17]*pca$feature17+eigen$V4[18]*pca$feature18+eigen$V4[19]*pca$feature19+eigen$V4[20]*pca$feature20+eigen$V4[21]*pca$feature21;
# pca$v5<-eigen$V5[1]*pca$feature1+eigen$V5[2]*pca$feature2+eigen$V5[3]*pca$feature3+eigen$V5[4]*pca$feature4+eigen$V5[5]*pca$feature5+eigen$V5[6]*pca$feature6+eigen$V5[7]*pca$feature7+eigen$V5[8]*pca$feature8+eigen$V5[9]*pca$feature9+eigen$V5[10]*pca$feature10+eigen$V5[11]*pca$feature11+eigen$V5[12]*pca$feature12+eigen$V5[13]*pca$feature13+eigen$V5[14]*pca$feature14+eigen$V5[15]*pca$feature15+eigen$V5[16]*pca$feature16+eigen$V5[17]*pca$feature17+eigen$V5[18]*pca$feature18+eigen$V5[19]*pca$feature19+eigen$V5[20]*pca$feature20+eigen$V5[21]*pca$feature21;
# pca$v6<-eigen$V6[1]*pca$feature1+eigen$V6[2]*pca$feature2+eigen$V6[3]*pca$feature3+eigen$V6[4]*pca$feature4+eigen$V6[5]*pca$feature5+eigen$V6[6]*pca$feature6+eigen$V6[7]*pca$feature7+eigen$V6[8]*pca$feature8+eigen$V6[9]*pca$feature9+eigen$V6[10]*pca$feature10+eigen$V6[11]*pca$feature11+eigen$V6[12]*pca$feature12+eigen$V6[13]*pca$feature13+eigen$V6[14]*pca$feature14+eigen$V6[15]*pca$feature15+eigen$V6[16]*pca$feature16+eigen$V6[17]*pca$feature17+eigen$V6[18]*pca$feature18+eigen$V6[19]*pca$feature19+eigen$V6[20]*pca$feature20+eigen$V6[21]*pca$feature21;
# pca$v7<-eigen$V7[1]*pca$feature1+eigen$V7[2]*pca$feature2+eigen$V7[3]*pca$feature3+eigen$V7[4]*pca$feature4+eigen$V7[5]*pca$feature5+eigen$V7[6]*pca$feature6+eigen$V7[7]*pca$feature7+eigen$V7[8]*pca$feature8+eigen$V7[9]*pca$feature9+eigen$V7[10]*pca$feature10+eigen$V7[11]*pca$feature11+eigen$V7[12]*pca$feature12+eigen$V7[13]*pca$feature13+eigen$V7[14]*pca$feature14+eigen$V7[15]*pca$feature15+eigen$V7[16]*pca$feature16+eigen$V7[17]*pca$feature17+eigen$V7[18]*pca$feature18+eigen$V7[19]*pca$feature19+eigen$V7[20]*pca$feature20+eigen$V7[21]*pca$feature21;
# pca$v8<-eigen$V8[1]*pca$feature1+eigen$V8[2]*pca$feature2+eigen$V8[3]*pca$feature3+eigen$V8[4]*pca$feature4+eigen$V8[5]*pca$feature5+eigen$V8[6]*pca$feature6+eigen$V8[7]*pca$feature7+eigen$V8[8]*pca$feature8+eigen$V8[9]*pca$feature9+eigen$V8[10]*pca$feature10+eigen$V8[11]*pca$feature11+eigen$V8[12]*pca$feature12+eigen$V8[13]*pca$feature13+eigen$V8[14]*pca$feature14+eigen$V8[15]*pca$feature15+eigen$V8[16]*pca$feature16+eigen$V8[17]*pca$feature17+eigen$V8[18]*pca$feature18+eigen$V8[19]*pca$feature19+eigen$V8[20]*pca$feature20+eigen$V8[21]*pca$feature21;
# pca$v9<-eigen$V9[1]*pca$feature1+eigen$V9[2]*pca$feature2+eigen$V9[3]*pca$feature3+eigen$V9[4]*pca$feature4+eigen$V9[5]*pca$feature5+eigen$V9[6]*pca$feature6+eigen$V9[7]*pca$feature7+eigen$V9[8]*pca$feature8+eigen$V9[9]*pca$feature9+eigen$V9[10]*pca$feature10+eigen$V9[11]*pca$feature11+eigen$V9[12]*pca$feature12+eigen$V9[13]*pca$feature13+eigen$V9[14]*pca$feature14+eigen$V9[15]*pca$feature15+eigen$V9[16]*pca$feature16+eigen$V9[17]*pca$feature17+eigen$V9[18]*pca$feature18+eigen$V9[19]*pca$feature19+eigen$V9[20]*pca$feature20+eigen$V9[21]*pca$feature21;
# pca$v10<-eigen$V10[1]*pca$feature1+eigen$V10[2]*pca$feature2+eigen$V10[3]*pca$feature3+eigen$V10[4]*pca$feature4+eigen$V10[5]*pca$feature5+eigen$V10[6]*pca$feature6+eigen$V10[7]*pca$feature7+eigen$V10[8]*pca$feature8+eigen$V10[9]*pca$feature9+eigen$V10[10]*pca$feature10+eigen$V10[11]*pca$feature11+eigen$V10[12]*pca$feature12+eigen$V10[13]*pca$feature13+eigen$V10[14]*pca$feature14+eigen$V10[15]*pca$feature15+eigen$V10[16]*pca$feature16+eigen$V10[17]*pca$feature17+eigen$V10[18]*pca$feature18+eigen$V10[19]*pca$feature19+eigen$V10[20]*pca$feature20+eigen$V10[21]*pca$feature21;
# pca$v11<-eigen$V11[1]*pca$feature1+eigen$V11[2]*pca$feature2+eigen$V11[3]*pca$feature3+eigen$V11[4]*pca$feature4+eigen$V11[5]*pca$feature5+eigen$V11[6]*pca$feature6+eigen$V11[7]*pca$feature7+eigen$V11[8]*pca$feature8+eigen$V11[9]*pca$feature9+eigen$V11[10]*pca$feature10+eigen$V11[11]*pca$feature11+eigen$V11[12]*pca$feature12+eigen$V11[13]*pca$feature13+eigen$V11[14]*pca$feature14+eigen$V11[15]*pca$feature15+eigen$V11[16]*pca$feature16+eigen$V11[17]*pca$feature17+eigen$V11[18]*pca$feature18+eigen$V11[19]*pca$feature19+eigen$V11[20]*pca$feature20+eigen$V11[21]*pca$feature21;
# pca$v12<-eigen$V12[1]*pca$feature1+eigen$V12[2]*pca$feature2+eigen$V12[3]*pca$feature3+eigen$V12[4]*pca$feature4+eigen$V12[5]*pca$feature5+eigen$V12[6]*pca$feature6+eigen$V12[7]*pca$feature7+eigen$V12[8]*pca$feature8+eigen$V12[9]*pca$feature9+eigen$V12[10]*pca$feature10+eigen$V12[11]*pca$feature11+eigen$V12[12]*pca$feature12+eigen$V12[13]*pca$feature13+eigen$V12[14]*pca$feature14+eigen$V12[15]*pca$feature15+eigen$V12[16]*pca$feature16+eigen$V12[17]*pca$feature17+eigen$V12[18]*pca$feature18+eigen$V12[19]*pca$feature19+eigen$V12[20]*pca$feature20+eigen$V12[21]*pca$feature21;
# pca$v13<-eigen$V13[1]*pca$feature1+eigen$V13[2]*pca$feature2+eigen$V13[3]*pca$feature3+eigen$V13[4]*pca$feature4+eigen$V13[5]*pca$feature5+eigen$V13[6]*pca$feature6+eigen$V13[7]*pca$feature7+eigen$V13[8]*pca$feature8+eigen$V13[9]*pca$feature9+eigen$V13[10]*pca$feature10+eigen$V13[11]*pca$feature11+eigen$V13[12]*pca$feature12+eigen$V13[13]*pca$feature13+eigen$V13[14]*pca$feature14+eigen$V13[15]*pca$feature15+eigen$V13[16]*pca$feature16+eigen$V13[17]*pca$feature17+eigen$V13[18]*pca$feature18+eigen$V13[19]*pca$feature19+eigen$V13[20]*pca$feature20+eigen$V13[21]*pca$feature21;
# 
# #switch nd and pca (uncomment)
# nd<-pca
# target<-nd$target


# Remove Features from Training Set
nd$target<-NULL

rm(trainM)
trainM<-data.matrix(nd, rownames.force = NA); #playing around with OHE
cat("Creating DMatrix for xgboost...\n");
dtrain <- xgb.DMatrix(data=trainM, label=target, missing = NaN);
watchlist <- list(trainM=dtrain);


################# Iterate 50-100 Times, Only need to do when changing sample size 
# Tuning Hyperparameters for XGBOOST Algorithm
# Automated to run overnight - this takes a long time.
# Use cross validated weeks 6-9 for Training.
#####################
nloops<-50
best_param = list()
best_seednumber = 1969
best_error = Inf
best_error_index = 0
library(mlr)
for (iter in 1:nloops) {
  param <- list(objective = "binary:logistic",
                booster= "gbtree",
                eval_metric="logloss",
                max_depth = sample(1:5, 1), #change to larger 
                eta = runif(1, .1, .3), #change from .1 to .05
                gamma = runif(1, 0.0, 0.2), #0.134885
                subsample = runif(1, .6, .9), #0.7742556
                colsample_bytree = runif(1, .5, .8), #0.5917445
                min_child_weight = sample(1:40, 1), #9
                max_delta_step = sample(1:10, 1) #change from 10 to 20
  )
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  cv.nround = 500
  cv.nfold = 5
  mdcv <- xgb.cv(data=dtrain, params = param, watchlist = watchlist, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=2, maximize=FALSE)
  
  min_error = min(mdcv$test.logloss.mean)
  min_error_index = which.min( mdcv$test.logloss.mean )
  
  if (min_error < best_error) {
    best_error = min_error
    best_error_index = min_error_index
    best_seednumber = seed.number
    best_param = param
  }
  cat("Loop:", iter,"\n");
}

nround = best_error_index
set.seed(best_seednumber)
cat("Best round:", nround,"\n");
cat("Best result:",best_error,"\n");
write.csv(data.frame(best_param), "~/Desktop/Data/Numerai/XGBPARAM10_26.csv", row.names = F)
############################################
# END XGBoost Tuning
############################################


######################################
# Train XGB Model
######################################
clf <- xgb.train(   params              = best_param, 
                    data                = dtrain, 
                    nrounds             = nround, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

##########################
# Test Variable Importance
##########################
importance <- xgb.importance(feature_names = colnames(trainM), model = clf)
xgb.plot.importance(importance)
print(importance)


##########################
# Begin Predictions on Tournament Data
# read in Competition data
##########################
cat("Loading data...\n");
numerai_tournament_data <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/numerai_tournament_data10_26.csv")
td<-data.table(numerai_tournament_data)

preds <- rep(0,nrow(td));
keep.id<-td$t_id
td$t_id<-NULL

# Make Predictions
testM <-data.matrix(td, rownames.force = NA)
preds <- round(predict(clf, testM),10)

####### THIS IS XGBOOST BASE PREDICTION SET ######################
numeraipred <- data.frame(id=keep.id, xgbp=preds);
#############################################################


cat("Adding feature data...\n");
rm(td)
td<-data.table(numerai_tournament_data)

td$r1_8<-td$feature1/td$feature8
td$r8_13<-td$feature8/td$feature13
td$r5_6<-td$feature5/td$feature6
td$r12_17<-td$feature12/td$feature17

# td$r1_12<-td$feature1/td$feature12
# td$r1_18<-td$feature1/td$feature18
# td$r2_17<-td$feature2/td$feature17
# td$r2_6<-td$feature2/td$feature6
# td$r2_7<-td$feature2/td$feature7
# td$r3_6<-td$feature3/td$feature6
# td$r3_14<-td$feature3/td$feature14
# td$r3_17<-td$feature3/td$feature17
# td$r4_9<-td$feature4/td$feature9
# td$r4_10<-td$feature4/td$feature10
# td$r5_8<-td$feature5/td$feature8
# td$r6_14<-td$feature6/td$feature14
# td$r6_17<-td$feature6/td$feature17
# td$r7_13<-td$feature7/td$feature13
# td$r7_17<-td$feature7/td$feature17
# td$r8_15<-td$feature8/td$feature15



# eigen vectors= comment out if do not use
#Eigenvectors from PCA
# rm(pca)
# pca<-td
# pca$v1<-eigen$V1[1]*pca$feature1+eigen$V1[2]*pca$feature2+eigen$V1[3]*pca$feature3+eigen$V1[4]*pca$feature4+eigen$V1[5]*pca$feature5+eigen$V1[6]*pca$feature6+eigen$V1[7]*pca$feature7+eigen$V1[8]*pca$feature8+eigen$V1[9]*pca$feature9+eigen$V1[10]*pca$feature10+eigen$V1[11]*pca$feature11+eigen$V1[12]*pca$feature12+eigen$V1[13]*pca$feature13+eigen$V1[14]*pca$feature14+eigen$V1[15]*pca$feature15+eigen$V1[16]*pca$feature16+eigen$V1[17]*pca$feature17+eigen$V1[18]*pca$feature18+eigen$V1[19]*pca$feature19+eigen$V1[20]*pca$feature20+eigen$V1[21]*pca$feature21;
# pca$v2<-eigen$V2[1]*pca$feature1+eigen$V2[2]*pca$feature2+eigen$V2[3]*pca$feature3+eigen$V2[4]*pca$feature4+eigen$V2[5]*pca$feature5+eigen$V2[6]*pca$feature6+eigen$V2[7]*pca$feature7+eigen$V2[8]*pca$feature8+eigen$V2[9]*pca$feature9+eigen$V2[10]*pca$feature10+eigen$V2[11]*pca$feature11+eigen$V2[12]*pca$feature12+eigen$V2[13]*pca$feature13+eigen$V2[14]*pca$feature14+eigen$V2[15]*pca$feature15+eigen$V2[16]*pca$feature16+eigen$V2[17]*pca$feature17+eigen$V2[18]*pca$feature18+eigen$V2[19]*pca$feature19+eigen$V2[20]*pca$feature20+eigen$V2[21]*pca$feature21;
# pca$v3<-eigen$V3[1]*pca$feature1+eigen$V3[2]*pca$feature2+eigen$V3[3]*pca$feature3+eigen$V3[4]*pca$feature4+eigen$V3[5]*pca$feature5+eigen$V3[6]*pca$feature6+eigen$V3[7]*pca$feature7+eigen$V3[8]*pca$feature8+eigen$V3[9]*pca$feature9+eigen$V3[10]*pca$feature10+eigen$V3[11]*pca$feature11+eigen$V3[12]*pca$feature12+eigen$V3[13]*pca$feature13+eigen$V3[14]*pca$feature14+eigen$V3[15]*pca$feature15+eigen$V3[16]*pca$feature16+eigen$V3[17]*pca$feature17+eigen$V3[18]*pca$feature18+eigen$V3[19]*pca$feature19+eigen$V3[20]*pca$feature20+eigen$V3[21]*pca$feature21;
# pca$v4<-eigen$V4[1]*pca$feature1+eigen$V4[2]*pca$feature2+eigen$V4[3]*pca$feature3+eigen$V4[4]*pca$feature4+eigen$V4[5]*pca$feature5+eigen$V4[6]*pca$feature6+eigen$V4[7]*pca$feature7+eigen$V4[8]*pca$feature8+eigen$V4[9]*pca$feature9+eigen$V4[10]*pca$feature10+eigen$V4[11]*pca$feature11+eigen$V4[12]*pca$feature12+eigen$V4[13]*pca$feature13+eigen$V4[14]*pca$feature14+eigen$V4[15]*pca$feature15+eigen$V4[16]*pca$feature16+eigen$V4[17]*pca$feature17+eigen$V4[18]*pca$feature18+eigen$V4[19]*pca$feature19+eigen$V4[20]*pca$feature20+eigen$V4[21]*pca$feature21;
# pca$v5<-eigen$V5[1]*pca$feature1+eigen$V5[2]*pca$feature2+eigen$V5[3]*pca$feature3+eigen$V5[4]*pca$feature4+eigen$V5[5]*pca$feature5+eigen$V5[6]*pca$feature6+eigen$V5[7]*pca$feature7+eigen$V5[8]*pca$feature8+eigen$V5[9]*pca$feature9+eigen$V5[10]*pca$feature10+eigen$V5[11]*pca$feature11+eigen$V5[12]*pca$feature12+eigen$V5[13]*pca$feature13+eigen$V5[14]*pca$feature14+eigen$V5[15]*pca$feature15+eigen$V5[16]*pca$feature16+eigen$V5[17]*pca$feature17+eigen$V5[18]*pca$feature18+eigen$V5[19]*pca$feature19+eigen$V5[20]*pca$feature20+eigen$V5[21]*pca$feature21;
# pca$v6<-eigen$V6[1]*pca$feature1+eigen$V6[2]*pca$feature2+eigen$V6[3]*pca$feature3+eigen$V6[4]*pca$feature4+eigen$V6[5]*pca$feature5+eigen$V6[6]*pca$feature6+eigen$V6[7]*pca$feature7+eigen$V6[8]*pca$feature8+eigen$V6[9]*pca$feature9+eigen$V6[10]*pca$feature10+eigen$V6[11]*pca$feature11+eigen$V6[12]*pca$feature12+eigen$V6[13]*pca$feature13+eigen$V6[14]*pca$feature14+eigen$V6[15]*pca$feature15+eigen$V6[16]*pca$feature16+eigen$V6[17]*pca$feature17+eigen$V6[18]*pca$feature18+eigen$V6[19]*pca$feature19+eigen$V6[20]*pca$feature20+eigen$V6[21]*pca$feature21;
# pca$v7<-eigen$V7[1]*pca$feature1+eigen$V7[2]*pca$feature2+eigen$V7[3]*pca$feature3+eigen$V7[4]*pca$feature4+eigen$V7[5]*pca$feature5+eigen$V7[6]*pca$feature6+eigen$V7[7]*pca$feature7+eigen$V7[8]*pca$feature8+eigen$V7[9]*pca$feature9+eigen$V7[10]*pca$feature10+eigen$V7[11]*pca$feature11+eigen$V7[12]*pca$feature12+eigen$V7[13]*pca$feature13+eigen$V7[14]*pca$feature14+eigen$V7[15]*pca$feature15+eigen$V7[16]*pca$feature16+eigen$V7[17]*pca$feature17+eigen$V7[18]*pca$feature18+eigen$V7[19]*pca$feature19+eigen$V7[20]*pca$feature20+eigen$V7[21]*pca$feature21;
# pca$v8<-eigen$V8[1]*pca$feature1+eigen$V8[2]*pca$feature2+eigen$V8[3]*pca$feature3+eigen$V8[4]*pca$feature4+eigen$V8[5]*pca$feature5+eigen$V8[6]*pca$feature6+eigen$V8[7]*pca$feature7+eigen$V8[8]*pca$feature8+eigen$V8[9]*pca$feature9+eigen$V8[10]*pca$feature10+eigen$V8[11]*pca$feature11+eigen$V8[12]*pca$feature12+eigen$V8[13]*pca$feature13+eigen$V8[14]*pca$feature14+eigen$V8[15]*pca$feature15+eigen$V8[16]*pca$feature16+eigen$V8[17]*pca$feature17+eigen$V8[18]*pca$feature18+eigen$V8[19]*pca$feature19+eigen$V8[20]*pca$feature20+eigen$V8[21]*pca$feature21;
# pca$v9<-eigen$V9[1]*pca$feature1+eigen$V9[2]*pca$feature2+eigen$V9[3]*pca$feature3+eigen$V9[4]*pca$feature4+eigen$V9[5]*pca$feature5+eigen$V9[6]*pca$feature6+eigen$V9[7]*pca$feature7+eigen$V9[8]*pca$feature8+eigen$V9[9]*pca$feature9+eigen$V9[10]*pca$feature10+eigen$V9[11]*pca$feature11+eigen$V9[12]*pca$feature12+eigen$V9[13]*pca$feature13+eigen$V9[14]*pca$feature14+eigen$V9[15]*pca$feature15+eigen$V9[16]*pca$feature16+eigen$V9[17]*pca$feature17+eigen$V9[18]*pca$feature18+eigen$V9[19]*pca$feature19+eigen$V9[20]*pca$feature20+eigen$V9[21]*pca$feature21;
# pca$v10<-eigen$V10[1]*pca$feature1+eigen$V10[2]*pca$feature2+eigen$V10[3]*pca$feature3+eigen$V10[4]*pca$feature4+eigen$V10[5]*pca$feature5+eigen$V10[6]*pca$feature6+eigen$V10[7]*pca$feature7+eigen$V10[8]*pca$feature8+eigen$V10[9]*pca$feature9+eigen$V10[10]*pca$feature10+eigen$V10[11]*pca$feature11+eigen$V10[12]*pca$feature12+eigen$V10[13]*pca$feature13+eigen$V10[14]*pca$feature14+eigen$V10[15]*pca$feature15+eigen$V10[16]*pca$feature16+eigen$V10[17]*pca$feature17+eigen$V10[18]*pca$feature18+eigen$V10[19]*pca$feature19+eigen$V10[20]*pca$feature20+eigen$V10[21]*pca$feature21;
# pca$v11<-eigen$V11[1]*pca$feature1+eigen$V11[2]*pca$feature2+eigen$V11[3]*pca$feature3+eigen$V11[4]*pca$feature4+eigen$V11[5]*pca$feature5+eigen$V11[6]*pca$feature6+eigen$V11[7]*pca$feature7+eigen$V11[8]*pca$feature8+eigen$V11[9]*pca$feature9+eigen$V11[10]*pca$feature10+eigen$V11[11]*pca$feature11+eigen$V11[12]*pca$feature12+eigen$V11[13]*pca$feature13+eigen$V11[14]*pca$feature14+eigen$V11[15]*pca$feature15+eigen$V11[16]*pca$feature16+eigen$V11[17]*pca$feature17+eigen$V11[18]*pca$feature18+eigen$V11[19]*pca$feature19+eigen$V11[20]*pca$feature20+eigen$V11[21]*pca$feature21;
# pca$v12<-eigen$V12[1]*pca$feature1+eigen$V12[2]*pca$feature2+eigen$V12[3]*pca$feature3+eigen$V12[4]*pca$feature4+eigen$V12[5]*pca$feature5+eigen$V12[6]*pca$feature6+eigen$V12[7]*pca$feature7+eigen$V12[8]*pca$feature8+eigen$V12[9]*pca$feature9+eigen$V12[10]*pca$feature10+eigen$V12[11]*pca$feature11+eigen$V12[12]*pca$feature12+eigen$V12[13]*pca$feature13+eigen$V12[14]*pca$feature14+eigen$V12[15]*pca$feature15+eigen$V12[16]*pca$feature16+eigen$V12[17]*pca$feature17+eigen$V12[18]*pca$feature18+eigen$V12[19]*pca$feature19+eigen$V12[20]*pca$feature20+eigen$V12[21]*pca$feature21;
# pca$v13<-eigen$V13[1]*pca$feature1+eigen$V13[2]*pca$feature2+eigen$V13[3]*pca$feature3+eigen$V13[4]*pca$feature4+eigen$V13[5]*pca$feature5+eigen$V13[6]*pca$feature6+eigen$V13[7]*pca$feature7+eigen$V13[8]*pca$feature8+eigen$V13[9]*pca$feature9+eigen$V13[10]*pca$feature10+eigen$V13[11]*pca$feature11+eigen$V13[12]*pca$feature12+eigen$V13[13]*pca$feature13+eigen$V13[14]*pca$feature14+eigen$V13[15]*pca$feature15+eigen$V13[16]*pca$feature16+eigen$V13[17]*pca$feature17+eigen$V13[18]*pca$feature18+eigen$V13[19]*pca$feature19+eigen$V13[20]*pca$feature20+eigen$V13[21]*pca$feature21;
# td<-pca
# rm(pca)


preds <- rep(0,nrow(td));
keep.id<-td$t_id
td$t_id<-NULL

# Make Predictions
testM <-data.matrix(td, rownames.force = NA)
preds <- round(predict(clf, testM),10)

####### THIS IS XGBOOST FEATURE PREDICTION SET ######################
numeraifeature <- data.frame(id=keep.id, feature=preds);
setkey(as.data.table(numeraifeature), "id")
write.csv(data.frame("t_id"=numeraifeature$id, "probability"=numeraifeature$feature), "~/Desktop/Data/Numerai/FEATUREADD.csv", row.names = F)
#############################################################
########## Completed with NUMERAI evaluation ################
#############################################################

############################
# BUILD ENSEMBLE
############################
ensemble<-NULL
ensemble<-merge(numeraipred, numeraifeature, by = c("id"), all.x = TRUE)
ensemble$minp<-apply(ensemble[,2:3],1,min)
ensemble$maxp<-apply(ensemble[,2:3],1,max)

#straight averages
ensemble$average<-(ensemble$feature+ensemble$xgbp)/2
ensemble$geom<-sqrt((ensemble$feature^2+ensemble$xgbp^2)/2)
ensemble$average.min<-(ensemble$feature+ensemble$xgbp-ensemble$minp)/2
ensemble$average.max<-(ensemble$feature+ensemble$xgbp-ensemble$maxp)/2

###############################
# write out ensembles
###############################
cat("Saving the submission files\n");
setkey(as.data.table(ensemble), "id")
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$xgbp), "~/Desktop/Data/Numerai/XGB10_26.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$feature), "~/Desktop/Data/Numerai/FEATURE10_26.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$average), "~/Desktop/Data/Numerai/AVERAGE10_26.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$geom), "~/Desktop/Data/Numerai/GEOM10_26.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$minp), "~/Desktop/Data/Numerai/MIN10_26.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$maxp), "~/Desktop/Data/Numerai/MAX10_26.csv", row.names = F)

