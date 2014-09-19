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
Clean up the variables by removing problematic fields, applying KNN imputation, removing overly correlated and near zero variance fields:

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
dfCleanTrain <- cbind(dfFactors, dfNumerics)
# add the classe variable back to the data set
dfCleanTrain$classe <- outcomeData
```

<BR><BR>
Shuffle the data...

```r
# shuffle the whole data set
set.seed(1234)
dfCleanTrain <- dfCleanTrain[sample(nrow(dfCleanTrain)), ]
```

<BR><BR>
With the training data set in a cleaned and ready state, we now model the entire data set with a <b>multinomial</b> model which can handle predicting multiple classes at a time, not just two. We also only use top 100 variables to accelerate modeling.

```r
library(nnet)
model = multinom(classe ~ ., data = dfCleanTrain, maxit = 500, trace = F)
# Get top 100 best variables
topModels <- varImp(model)
topModels$Variables <- row.names(topModels)
topModels <- topModels[order(-topModels$Overall), ]
topVariables <- head(topModels, 100)$Variables
```

<BR><BR>
Here is the list of the top 10 variables ranked as most important by the varImp function:

```r
print(head(topVariables, 10))
```

```
##  [1] "avg_yaw_belt"           "min_roll_belt"         
##  [3] "max_roll_belt"          "stddev_yaw_belt"       
##  [5] "max_roll_forearm"       "amplitude_roll_forearm"
##  [7] "amplitude_roll_belt"    "var_yaw_belt"          
##  [9] "min_roll_forearm"       "roll_belt"
```

<BR><BR>
We now cross-validate the <b>multinomial</b> model using a different 5th of the training data set as a test data set to get an more correct average error and accuracy score.

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
    objModel <- multinom(as.formula(f), data = dataTrain, maxit = 500, trace = F)
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
## [1] 0.8199
```

The error rate derived from the accuracy minus 1 (or directly from the metrics package function: ce):

```r
print(mean(totalError))
```

```
## [1] 0.1801
```

Finally we use the real test data set to predict the 20 cases using our model. We first must apply the top 100 variables to the testing data set so that the training and testing sets are on the same page:

```r
f <- paste("classe ~ ", paste0(topVariables, collapse = " + "))
objModel <- multinom(as.formula(f), data = dfCleanTrain, maxit = 500, trace = F)
dfCleanTest <- dfTest[c(intersect(names(dfCleanTrain), names(dfTest)))]
dfCleanTest$new_window <- as.numeric(dfCleanTest$new_window)
# impute all NAs to 0
dfCleanTest[is.na(dfCleanTest)] <- 0
preds <- predict(objModel, type = "class", newdata = dfCleanTest)
preds
```

```
##  [1] B A A A A C C C A A A A B A C A A B B A
## Levels: A B C D E
```

