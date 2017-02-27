
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


# Select which Source File
nd<-data.table(numerai_training_data)

# Retain Target Prediction
rm(target)
target<-nd$target

###########################
# Add Features To Training
###########################

# Create Tables of Unique Feature IDs
cat1<-NULL
cat1<-unique(data.frame(nd)[,'feature1',drop = FALSE])
cat1$cat1_id<-seq.int(nrow(cat1))
setkey(nd, feature1)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature1)]
setnames(f_target,"V1","f1_t_mean")
cat1 <- merge(cat1, f_target, by = c("feature1"), all.x = TRUE)

cat2<-unique(data.frame(nd)[,'feature2',drop = FALSE])
cat2$cat2_id<-seq.int(nrow(cat2))
setkey(nd, feature2)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature2)]
setnames(f_target,"V1","f2_t_mean")
cat2 <- merge(cat2, f_target, by = c("feature2"), all.x = TRUE)

cat3<-unique(data.frame(nd)[,'feature3',drop = FALSE])
cat3$cat3_id<-seq.int(nrow(cat3))
setkey(nd, feature3)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature3)]
setnames(f_target,"V1","f3_t_mean")
cat3 <- merge(cat3, f_target, by = c("feature3"), all.x = TRUE)

cat4<-unique(data.frame(nd)[,'feature4',drop = FALSE])
cat4$cat4_id<-seq.int(nrow(cat4))
setkey(nd, feature4)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature4)]
setnames(f_target,"V1","f4_t_mean")
cat4 <- merge(cat4, f_target, by = c("feature4"), all.x = TRUE)

cat5<-unique(data.frame(nd)[,'feature5',drop = FALSE])
cat5$cat5_id<-seq.int(nrow(cat5))
setkey(nd, feature5)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature5)]
setnames(f_target,"V1","f5_t_mean")
cat5 <- merge(cat5, f_target, by = c("feature5"), all.x = TRUE)

cat6<-unique(data.frame(nd)[,'feature6',drop = FALSE])
cat6$cat6_id<-seq.int(nrow(cat6))
setkey(nd, feature6)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature6)]
setnames(f_target,"V1","f6_t_mean")
cat6 <- merge(cat6, f_target, by = c("feature6"), all.x = TRUE)

cat7<-unique(data.frame(nd)[,'feature7',drop = FALSE])
cat7$cat7_id<-seq.int(nrow(cat7))
setkey(nd, feature7)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature7)]
setnames(f_target,"V1","f7_t_mean")
cat7 <- merge(cat7, f_target, by = c("feature7"), all.x = TRUE)

cat8<-unique(data.frame(nd)[,'feature8',drop = FALSE])
cat8$cat8_id<-seq.int(nrow(cat8))
setkey(nd, feature8)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature8)]
setnames(f_target,"V1","f8_t_mean")
cat8 <- merge(cat8, f_target, by = c("feature8"), all.x = TRUE)

cat9<-unique(data.frame(nd)[,'feature9',drop = FALSE])
cat9$cat9_id<-seq.int(nrow(cat9))
setkey(nd, feature9)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature9)]
setnames(f_target,"V1","f9_t_mean")
cat9 <- merge(cat9, f_target, by = c("feature9"), all.x = TRUE)

cat10<-unique(data.frame(nd)[,'feature10',drop = FALSE])
cat10$cat10_id<-seq.int(nrow(cat10))
setkey(nd, feature10)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature10)]
setnames(f_target,"V1","f10_t_mean")
cat10 <- merge(cat10, f_target, by = c("feature10"), all.x = TRUE)

cat11<-unique(data.frame(nd)[,'feature11',drop = FALSE])
cat11$cat11_id<-seq.int(nrow(cat11))
setkey(nd, feature11)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature11)]
setnames(f_target,"V1","f11_t_mean")
cat11 <- merge(cat11, f_target, by = c("feature11"), all.x = TRUE)

cat12<-unique(data.frame(nd)[,'feature12',drop = FALSE])
cat12$cat12_id<-seq.int(nrow(cat12))
setkey(nd, feature12)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature12)]
setnames(f_target,"V1","f12_t_mean")
cat12 <- merge(cat12, f_target, by = c("feature12"), all.x = TRUE)

cat13<-unique(data.frame(nd)[,'feature13',drop = FALSE])
cat13$cat13_id<-seq.int(nrow(cat13))
setkey(nd, feature13)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature13)]
setnames(f_target,"V1","f13_t_mean")
cat13 <- merge(cat13, f_target, by = c("feature13"), all.x = TRUE)

cat14<-unique(data.frame(nd)[,'feature14',drop = FALSE])
cat14$cat14_id<-seq.int(nrow(cat14))
setkey(nd, feature14)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature14)]
setnames(f_target,"V1","f14_t_mean")
cat14 <- merge(cat14, f_target, by = c("feature14"), all.x = TRUE)

cat15<-unique(data.frame(nd)[,'feature15',drop = FALSE])
cat15$cat15_id<-seq.int(nrow(cat15))
setkey(nd, feature15)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature15)]
setnames(f_target,"V1","f15_t_mean")
cat15 <- merge(cat15, f_target, by = c("feature15"), all.x = TRUE)

cat16<-unique(data.frame(nd)[,'feature16',drop = FALSE])
cat16$cat16_id<-seq.int(nrow(cat16))
setkey(nd, feature16)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature16)]
setnames(f_target,"V1","f16_t_mean")
cat16 <- merge(cat16, f_target, by = c("feature16"), all.x = TRUE)

cat17<-unique(data.frame(nd)[,'feature17',drop = FALSE])
cat17$cat17_id<-seq.int(nrow(cat17))
setkey(nd, feature17)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature17)]
setnames(f_target,"V1","f17_t_mean")
cat17 <- merge(cat17, f_target, by = c("feature17"), all.x = TRUE)

cat18<-unique(data.frame(nd)[,'feature18',drop = FALSE])
cat18$cat18_id<-seq.int(nrow(cat18))
setkey(nd, feature18)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature18)]
setnames(f_target,"V1","f18_t_mean")
cat18 <- merge(cat18, f_target, by = c("feature18"), all.x = TRUE)

cat19<-unique(data.frame(nd)[,'feature19',drop = FALSE])
cat19$cat19_id<-seq.int(nrow(cat19))
setkey(nd, feature19)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature19)]
setnames(f_target,"V1","f19_t_mean")
cat19 <- merge(cat19, f_target, by = c("feature19"), all.x = TRUE)

cat20<-unique(data.frame(nd)[,'feature20',drop = FALSE])
cat20$cat20_id<-seq.int(nrow(cat20))
setkey(nd, feature20)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature20)]
setnames(f_target,"V1","f20_t_mean")
cat20 <- merge(cat20, f_target, by = c("feature20"), all.x = TRUE)

cat21<-unique(data.frame(nd)[,'feature21',drop = FALSE])
cat21$cat21_id<-seq.int(nrow(cat21))
setkey(nd, feature21)
f_target<-NULL
f_target <- nd[, mean(target),by = .(feature21)]
setnames(f_target,"V1","f21_t_mean")
cat21 <- merge(cat21, f_target, by = c("feature21"), all.x = TRUE)

# Remove target field from Training Set and reset to original source order
nd<-data.table(numerai_training_data)
nd$target<-NULL


###########################################
# Add TEST File Rows before adding features
###########################################
cat("Loading data...\n");
numerai_tournament_data <- read.csv("~/Desktop/Data/Numerai/numerai_datasets/numerai_tournament_data10_26.csv")
td<-data.table(numerai_tournament_data)

# keep track of rows and id's
keep.id<-td$t_id
test.rows<-nrow(td)
train.rows<-nrow(nd)
td$t_id<-NULL

# merge train and test
nd<-bind_rows(nd, td)
nd$origin_id<-seq.int(nrow(nd))

# merge category target means fields
#setkey(nd, feature1)
nd <- merge(nd,cat1, by = c("feature1"), all.x = TRUE)
#setkey(nd, feature2)
nd <- merge(nd,cat2, by = c("feature2"), all.x = TRUE)
#setkey(nd, feature3)
nd <- merge(nd,cat3, by = c("feature3"), all.x = TRUE)
#setkey(nd, feature4)
nd <- merge(nd,cat4, by = c("feature4"), all.x = TRUE)
#setkey(nd, feature5)
nd <- merge(nd,cat5, by = c("feature5"), all.x = TRUE)
#setkey(nd, feature6)
nd <- merge(nd,cat6, by = c("feature6"), all.x = TRUE)
#setkey(nd, feature7)
nd <- merge(nd,cat7, by = c("feature7"), all.x = TRUE)
#setkey(nd, feature8)
nd <- merge(nd,cat8, by = c("feature8"), all.x = TRUE)
#setkey(nd, feature9)
nd <- merge(nd,cat9, by = c("feature9"), all.x = TRUE)
#setkey(nd, feature10)
nd <- merge(nd,cat10, by = c("feature10"), all.x = TRUE)
#setkey(nd, feature11)
nd <- merge(nd,cat11, by = c("feature11"), all.x = TRUE)
#setkey(nd, feature12)
nd <- merge(nd,cat12, by = c("feature12"), all.x = TRUE)
#setkey(nd, feature13)
nd <- merge(nd,cat13, by = c("feature13"), all.x = TRUE)
#setkey(nd, feature14)
nd <- merge(nd,cat14, by = c("feature14"), all.x = TRUE)
#setkey(nd, feature15)
nd <- merge(nd,cat15, by = c("feature15"), all.x = TRUE)
#setkey(nd, feature16)
nd <- merge(nd,cat16, by = c("feature16"), all.x = TRUE)
#setkey(nd, feature17)
nd <- merge(nd,cat17, by = c("feature17"), all.x = TRUE)
#setkey(nd, feature18)
nd <- merge(nd,cat18, by = c("feature18"), all.x = TRUE)
#setkey(nd, feature19)
nd <- merge(nd,cat19, by = c("feature19"), all.x = TRUE)
#setkey(nd, feature20)
nd <- merge(nd,cat20, by = c("feature20"), all.x = TRUE)
#setkey(nd, feature21)
nd <- merge(nd,cat21, by = c("feature21"), all.x = TRUE)

# Add feature ratios
nd$r1_8<-nd$feature1/nd$feature8
nd$r8_13<-nd$feature8/nd$feature13
nd$r5_6<-nd$feature5/nd$feature6
nd$r12_17<-nd$feature12/nd$feature17

nd$r1_12<-nd$feature1/nd$feature12
nd$r1_18<-nd$feature1/nd$feature18
nd$r2_17<-nd$feature2/nd$feature17
nd$r2_6<-nd$feature2/nd$feature6
nd$r2_7<-nd$feature2/nd$feature7
nd$r3_6<-nd$feature3/nd$feature6
nd$r3_14<-nd$feature3/nd$feature14
nd$r3_17<-nd$feature3/nd$feature17
nd$r4_9<-nd$feature4/nd$feature9
nd$r4_10<-nd$feature4/nd$feature10
nd$r5_8<-nd$feature5/nd$feature8
nd$r6_14<-nd$feature6/nd$feature14
nd$r6_17<-nd$feature6/nd$feature17
nd$r7_13<-nd$feature7/nd$feature13
nd$r7_17<-nd$feature7/nd$feature17
nd$r8_15<-nd$feature8/nd$feature15

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

# Separate Train and Test now that Features are added
setkey(nd,origin_id)
max.rows<-nrow(nd)
nd$origin_id<-NULL
# TEST
td<-nd[(train.rows+1):max.rows]

# TRAIN
nd<-nd[1:train.rows]
######################################
######################################
# Added all Features to TEST and TRAIN
# Separated TEST and TRAIN
######################################
######################################


######################################
# Remove Any Features deemed unworthy by selecting training and testing set
######################################
sparse_matrix_train<-NULL
sparse_matrix_train <- sparse.model.matrix(~cat1_id+cat2_id+cat3_id+cat4_id+cat5_id+cat6_id
                                           +cat7_id+cat8_id+cat9_id+cat10_id+cat11_id
                                           +cat12_id+cat13_id+cat14_id+cat15_id+cat16_id
                                           +cat17_id+cat18_id+cat19_id+cat20_id+cat21_id -1, data = nd)
merge_with_ohe<-NULL

# copy this line and then select the fields you want to keep
# merge_with_ohe<-select(nd, feature1, feature2, feature3, feature4, feature5, feature6,
#                               feature7, feature8, feature9, feature10, feature11, feature12,
#                               feature13, feature14, feature15, feature16, feature17, feature18,
#                               feature19, feature20, feature21, r1_8, r8_13, r5_6, r12_17,
#                               f1_t_mean, f2_t_mean, f3_t_mean, f4_t_mean, f5_t_mean, f6_t_mean,
#                               f7_t_mean, f8_t_mean, f9_t_mean, f10_t_mean, f11_t_mean, f12_t_mean,
#                               f13_t_mean, f14_t_mean, f15_t_mean, f16_t_mean, f17_t_mean, f18_t_mean,
#                               f19_t_mean, f20_t_mean, f21_t_mean)

merge_with_ohe<-select(nd, feature1, feature2, feature3, feature4, feature5, feature6,
                              feature7, feature8, feature9, feature10, feature11, feature12,
                              feature13, feature14, feature15, feature16, feature17, feature18,
                              feature19, feature20, feature21, r1_8, r8_13, r5_6, r12_17
                              )

merge_with_ohe<-data.matrix(merge_with_ohe)
sparse_matrix_train<-cbind2(sparse_matrix_train,merge_with_ohe)

rm(trainM)
trainM<-data.matrix(sparse_matrix_train, rownames.force = NA); #playing around with OHE
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
                max_depth = sample(1:5, 1), #8 
                eta = runif(1, .1, .3), #changed from .1 to .05
                gamma = runif(1, 0.0, 0.2), #0.134885
                subsample = runif(1, .6, .9), #0.7742556
                colsample_bytree = runif(1, .5, .8), #0.5917445
                min_child_weight = sample(1:40, 1), #9
                max_delta_step = sample(1:10, 1) #changed from 10 to 20
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
write.csv(data.frame(best_param), "~/Desktop/Data/Numerai/XGBPARAMOHE.csv", row.names = F)
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
preds <- rep(0,nrow(td));

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
write.csv(data.frame("t_id"=numeraifeature$id, "probability"=numeraifeature$feature), "~/Desktop/Data/Numerai/FEATUREOHE.csv", row.names = F)
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
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$xgbp), "~/Desktop/Data/Numerai/XGBohe.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$feature), "~/Desktop/Data/Numerai/FEATUREohe.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$average), "~/Desktop/Data/Numerai/AVERAGEohe.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$geom), "~/Desktop/Data/Numerai/GEOMohe.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$minp), "~/Desktop/Data/Numerai/MINohe.csv", row.names = F)
write.csv(data.frame("t_id"=ensemble$id, "probability"=ensemble$maxp), "~/Desktop/Data/Numerai/MAXohe.csv", row.names = F)

