Practical Machine Learning Project
========================================================

Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to predict the manner in which they did the exercise. 
<BR><BR>
Getting the data online once and subsequently loading it from a local directory:

```r
setwd("/Volumes/NO NAME/classes/Hopkins/Practical Machine Learning/")
library(data.table)
library(RCurl)
```

```
## Loading required package: bitops
```

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r

# urlfile
# <-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
# download.file(urlfile, destfile = 'temp.csv', method = 'curl') dfTrain
# <-read.csv('temp.csv') write.csv(dfTrain, 'train.csv',row.names=F)
dfTrain <- read.csv("train.csv")

# urlfile
# <-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
# download.file(urlfile, destfile = 'temp.csv', method = 'curl') dfTest
# <-read.csv('temp.csv') write.csv(dfTest, 'test.csv',row.names=F)
dfTest <- read.csv("test.csv")
```

<BR><BR>
Clean up the variables by removing problematic fields and imputating all NAs with either KNN or 0:

```r
# remove index, un-important features, and timestamps
dfTrain <- subset(dfTrain, select = -c(X, user_name, cvtd_timestamp, raw_timestamp_part_1, 
    raw_timestamp_part_2))
# remove the outcome temporarily
outcomeData <- dfTrain$classe
dfTrain <- subset(dfTrain, select = -c(classe))
# get names of numeric data
originalNumericVariables <- names(dfTrain[sapply(dfTrain, is.numeric)])
# get names of factor data
originalFactorVariables <- names(dfTrain[sapply(dfTrain, is.factor)])
# binarize all numerical factors to numericals
dfFactors <- as.data.frame(lapply(dfTrain[c(originalFactorVariables)], as.numeric))
# impute numerics with nearest knn
dfNumerics <- dfTrain[originalNumericVariables]
imputedNums <- preProcess(dfNumerics, method = c("knnImpute"))
dfNumerics <- predict(imputedNums, dfNumerics)
# rebuild data frame
dfCleanTrain <- cbind(dfFactors, dfNumerics)
# remove near zero variance
zeroVar <- nearZeroVar(dfCleanTrain)
dfCleanTrain <- dfCleanTrain[-zeroVar]
# impute any other NAs with 0
dfCleanTrain[is.na(dfCleanTrain)] <- 0
# add the classe variable back to the data set
dfCleanTrain$classe <- outcomeData
```

<BR><BR>
Clean up testing data set 

```r
# clean up test data set
dfCleanTest <- dfTest[c(intersect(names(dfCleanTrain), names(dfTest)))]
# get names of numeric data
originalNumericVariables <- names(dfCleanTest[sapply(dfCleanTest, is.numeric)])
# get names of factor data
originalFactorVariables <- names(dfCleanTest[sapply(dfCleanTest, is.logical)])
# binarize all numerical factors to numericals
dfFactors <- as.data.frame(lapply(dfCleanTest[c(originalFactorVariables)], as.numeric))
# impute numerics with nearest knn
dfNumerics <- dfCleanTest[originalNumericVariables]
imputedNums <- preProcess(dfNumerics, method = c("knnImpute"))
dfNumerics <- predict(imputedNums, dfNumerics)
# rebuild data frame
dfCleanTest <- cbind(dfFactors, dfNumerics)
# impute any other NAs with 0
dfCleanTest[is.na(dfCleanTest)] <- 0
# remove near zero variance
zeroVar <- nearZeroVar(dfCleanTest)
dfCleanTest <- dfCleanTest[-zeroVar]
```

<BR><BR>
Balance out both data sets to only use mutually inclusive features

```r
# balance both train and test datasets
predictors <- intersect(names(dfCleanTrain), names(dfCleanTest))
dfCleanTrain <- dfCleanTrain[c(predictors, "classe")]
dfCleanTest <- dfCleanTest[c(predictors)]
```

<BR><BR>
Shuffle the data...

```r
# shuffle the whole data set
set.seed(1234)
dfCleanTrain <- dfCleanTrain[sample(nrow(dfCleanTrain)), ]
```

<BR><BR>
With the training data set in a cleaned and ready state, we now model the entire data set with a <b>multinomial</b> model which can handle predicting multiple classes at a time, not just two.

```r
library(nnet)
model = multinom(classe ~ ., data = dfCleanTrain, maxit = 1000, trace = F)
# Order top influential variables
topModels <- varImp(model)
topModels$Variables <- row.names(topModels)
topModels <- topModels[order(-topModels$Overall), ]
topVariables <- head(topModels, 120)$Variables
```

<BR><BR>
Here is the list of the top 10 variables ranked as most important by the varImp function:

```r
print(head(topVariables, 10))
```

```
##  [1] "yaw_belt"          "roll_belt"         "pitch_belt"       
##  [4] "magnet_dumbbell_z" "magnet_belt_z"     "magnet_arm_z"     
##  [7] "accel_belt_z"      "accel_arm_z"       "magnet_dumbbell_x"
## [10] "magnet_dumbbell_y"
```

<BR><BR>
We now cross-validate the <b>multinomial</b> model using a different portion of the training data set as a test data set to get an more correct average error and accuracy score.

```r
f <- paste("classe ~ ", paste0(topVariables, collapse = " + "))
# cross validate cv times
cv <- 3
cvDivider <- floor(nrow(dfCleanTrain)/(cv + 1))
indexCount <- 1

# use multinom model from nnet to predict all classes use metrics RMSLE
# function to averate teh root mean squared log error from all runs
library(Metrics)
totalError <- c()
totalAccuracy <- c()
for (cv in seq(1:cv)) {
    # assign chunk to data test
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfCleanTrain[dataTestIndex, ]
    # everything else to train
    dataTrain <- dfCleanTrain[-dataTestIndex, ]
    objModel <- multinom(as.formula(f), data = dataTrain, maxit = 1000, trace = F)
    preds <- predict(objModel, type = "class", newdata = dataTest)
    err <- ce(as.numeric(dataTest$classe), as.numeric(preds))
    totalError <- c(totalError, err)
    totalAccuracy <- c(totalAccuracy, postResample(dataTest$classe, preds)[[1]])
}
```

<BR><BR>
The accuracy of the model is found using postResample and returns:

```r
print(mean(totalAccuracy))
```

```
## [1] 0.7471
```

<BR><BR>
The error rate derived from the accuracy minus 1 (or directly from the metrics package function: ce):

```r
print(mean(totalError))
```

```
## [1] 0.2529
```

<BR><BR>
Finally we use the real test data set to predict the 20 cases using our model:

```r
f <- paste("classe ~ ", paste0(topVariables, collapse = " + "))
objModel <- multinom(as.formula(f), data = dfCleanTrain, maxit = 1000)
```

```
## # weights:  275 (216 variable)
## initial  value 31580.390718 
## iter  10 value 22200.713920
## iter  20 value 19859.805716
## iter  30 value 18608.659746
## iter  40 value 17911.799810
## iter  50 value 17509.020539
## iter  60 value 16820.756778
## iter  70 value 15611.111940
## iter  80 value 14834.548165
## iter  90 value 14187.707443
## iter 100 value 13894.123783
## iter 110 value 13753.009687
## iter 120 value 13557.017079
## iter 130 value 13369.396716
## iter 140 value 13266.247995
## iter 150 value 13212.667029
## iter 160 value 13183.985695
## iter 170 value 13165.528062
## iter 180 value 13159.059592
## iter 190 value 13157.600624
## iter 200 value 13157.215013
## iter 210 value 13157.135004
## iter 220 value 13157.082552
## iter 230 value 13157.031470
## iter 240 value 13157.017313
## iter 240 value 13157.017248
## iter 240 value 13157.017247
## final  value 13157.017247 
## converged
```

```r
preds <- predict(objModel, type = "class", newdata = dfCleanTest)
preds
```

```
##  [1] A B B A A E D B B A D C B A B A A B A B
## Levels: A B C D E
```

