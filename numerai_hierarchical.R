
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

# read in Training data
cat("Loading data...\n");
numerai_training_data <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/numerai_training_data9_28.csv")

# Select which Source File
hier_nd<-data.table(numerai_training_data)

#global mean/median (+modification)
global <- hier_nd[, mean(target)]

#table of feature 1,3,... mean 
setkey(hier_nd, feature1, feature2, feature3)
mean_Prod_Client_Agent <- hier_train[, mean(log1p(Demanda_uni_equil)),by = .(Producto_ID,Cliente_ID,Agencia_ID)]
setnames(mean_Prod_Client_Agent,"V1","PCA")

#table of product -client -route mean 
setkey(hier_train, Producto_ID, Cliente_ID, Ruta_SAK)
mean_Prod_Client_Ruta <- hier_train[, mean(log1p(Demanda_uni_equil)),by = .(Producto_ID,Cliente_ID,Ruta_SAK)]
setnames(mean_Prod_Client_Ruta,"V1","PCR")

#table of product -route mean
setkey(hier_train, Producto_ID, Ruta_SAK)
mean_Prod_Ruta <- hier_train[, mean(log1p(Demanda_uni_equil)),by = .(Producto_ID, Ruta_SAK)]
setnames(mean_Prod_Ruta,"V1","PR")

#table of product -agency mean
setkey(hier_train, Producto_ID, Agencia_ID)
mean_Prod_Agency <- hier_train[, mean(log1p(Demanda_uni_equil)),by = .(Producto_ID, Agencia_ID)]
setnames(mean_Prod_Agency,"V1","PA")

#table of product overall mean 
mean_Prod <- hier_train[, mean(log1p(Demanda_uni_equil)), by = .(Producto_ID)]
setnames(mean_Prod,"V1","P")

#table of client overall mean 
setkey(hier_train, Cliente_ID)
mean_Client <- hier_train[, mean(log1p(Demanda_uni_equil)), by = .(Cliente_ID)]
setnames(mean_Client,"V1","C")

# add columns PCA, PCR, PR, PA, P, C, global for mean 
hold<-NULL
hold <- merge(hier_test, mean_Prod_Client_Agent, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Client_Ruta, by = c("Producto_ID", "Cliente_ID", "Ruta_SAK"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Ruta, by = c("Producto_ID", "Ruta_SAK"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Agency, by = c("Producto_ID", "Agencia_ID"), all.x = TRUE)
hold <- merge(hold, mean_Prod, by = "Producto_ID", all.x = TRUE)
hold <- merge(hold, mean_Client, by = "Cliente_ID", all.x = TRUE)











# Retain Target Prediction
rm(target)
target<-nd$target

# Add Features
#nd$featmin<-apply(nd[,],1,min)
#nd$featmax<-apply(nd[,],1,max)
#nd$featmean<-apply(nd[,],1,mean)
nd$r1_8<-nd$feature1/nd$feature8

nd$r8_13<-nd$feature8/nd$feature13
nd$r5_6<-nd$feature5/nd$feature6
nd$r12_17<-nd$feature12/nd$feature17
#nd$r12_18<-nd$feature12/nd$feature18


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
                max_depth = sample(8:15, 1), #8 
                eta = runif(1, .1, .3), #0.2784 
                gamma = runif(1, 0.0, 0.2), #0.134885
                subsample = runif(1, .6, .9), #0.7742556
                colsample_bytree = runif(1, .5, .8), #0.5917445
                min_child_weight = sample(1:40, 1), #9
                max_delta_step = sample(1:10, 1) #4
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
write.csv(data.frame(best_param), "~/Desktop/Data/Numerai/XGBPARAM9_28.csv", row.names = F)
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
numerai_tournament_data <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/numerai_tournament_data9_28.csv")
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
#td$r12_18<-td$feature12/td$feature18

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
########## Completed with NUMERAI evaluation #################
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
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$xgbp), "~/Desktop/Data/Numerai/XGB9_28.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$feature), "~/Desktop/Data/Numerai/FEATURE9_28.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$average), "~/Desktop/Data/Numerai/AVERAGE9_28.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$geom), "~/Desktop/Data/Numerai/GEOM9_28.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$minp), "~/Desktop/Data/Numerai/MIN9_28.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$maxp), "~/Desktop/Data/Numerai/MAX9_28.csv", row.names = F)

