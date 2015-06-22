---
title: "Practical Machine Learning Course Project"
author: "Roshan Riazi"
output: html_document
---

#Synopsis

In this report, we want to use [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) dataset and build a model to predict activity type, based on data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. After reading the dataset and exploring it a little, we splited the data into training, testing and validation. Then we removed some unnecessary features. In the next step, we trained three models ("rf", "gbm" and "lda") on training dataset and evaluated them on training and testing datasets, using accuracy metric. So, using these evaluations, we selected the best approach for prediction and evaluated it on validation dataset, which had 0.9941 accuracy. Finally we predicted the 20 final test observations using this final model.

#Background

This is the course project for [Practical Machine Learning course](https://class.coursera.org/predmachlearn-015) from Johns Hopkins University on Coursera.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

#Data

The data for this project comes from [Practical Machine Learning course](https://class.coursera.org/predmachlearn-015) from Johns Hopkins University on Coursera. We have [training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and a [final test](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) datasets.

First we check for a data folder and if there isn't create one. Then we check for training and test files and download them if they aren't in the data folder.


```r
fileTrainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileTestURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if(!dir.exists("./data")){dir.create("./data")}
if(!file.exists("./data/pml-testing.csv")){
    downloadDate <- date()
    download.file(fileTrainURL, destfile = "./data/pml-training.csv", method = "curl")
}
if(!file.exists("./data/pml-training.csv")){
    downloadDate <- date()
    download.file(fileTestURL, destfile = "./data/pml-testing.csv", method = "curl")
}
```

Then we read the training data in a variable named "FullData".


```r
FullData <- read.csv("./data/pml-training.csv")
dim(FullData)
```

```
## [1] 19622   160
```

We also done some exploratory analysis which we will not show, so that our report will not get too long.

##Data Partitioning

As our data had a large number of 19622 rows, we decided to split it into three parts: 60% training, 20% testing and 20% validation. We will use training to make our models, testing to make some evaluations and apply our final model to the validation set to make an approximation about out of sample error.


```r
#using caret
library(caret)
#partitioning data - 60, 20, 20
set.seed(1)
inTrain <- createDataPartition(FullData$classe, p = .8, list = FALSE)
validation <- FullData[-inTrain, ]
trainFull <- FullData[inTrain, ]
inTrain <- createDataPartition(trainFull$classe, p = .75, list = FALSE)
training <- trainFull[inTrain, ]
testing <- trainFull[-inTrain, ]
rm(trainFull)
```

#Preprocessing

Our preprocessing step was mainly feature selection and we didn't make any transformation on variables. First we remove some bookkeeping variables, and then changed facotr variables to numeric.


```r
#removing unrelated variables
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
validation <- validation[, -c(1:7)]
#str(training, list.len = 153)
#changing factors to numeric
for(i in 1:152){
    if(class(training[, i]) == "factor"){
        training[, i] <- as.numeric(as.factor(training[, i]))
        testing[, i] <- as.numeric(as.factor(testing[, i]))
        validation[, i] <- as.numeric(as.factor(validation[, i]))
    }
}
#str(FullDataUsefull, list.len = 153)
#summary(FullData$classe)
```

Then we tried to narrow these 153 variables.

##Feature Selection

One of the most common functions for feature removal is nearZeroVar. This function find the variables that have a variance near zero (one dominant value and low ratio of distinct values.)


```r
#removing near zero variance variables
nzv <- nearZeroVar(training[, -153])
length(nzv)
```

```
## [1] 52
```

```r
trainingFiltered <- training[, -nzv]
testingFiltered <- testing[, -nzv]
validationFiltered <- validation[, -nzv]
dim(trainingFiltered)
```

```
## [1] 11776   101
```

Also by looking at variables, we found some variables that had a large (in some cases more that 97%) NA values and removed them.


```r
#checking for redundant variables
excludeCols <- grep("^max|^min|^amplitude|^var|^avg|^stddev", x = names(trainingFiltered))
trainingEx <- trainingFiltered[, -excludeCols]
testingEx <- testingFiltered[, -excludeCols]
validationEx <- validationFiltered[, -excludeCols]
dim(trainingEx)
```

```
## [1] 11776    53
```

Then, using "findCorrelation" function, we looked for variables which had a high (more than 95%) correlation with other variables and removed them.


```r
#removing correlated variables with cutoff = 0.95
corVars <- findCorrelation(x = cor(trainingEx[, -53]), cutoff = 0.95)
trainingExCor <- trainingEx[, -corVars]
testingExCor <- testingEx[, -corVars]
validationExCor <- validationEx[, -corVars]
dim(trainingExCor)
```

```
## [1] 11776    48
```

With these techniques, we reduced the number of variables from 160 to 48.

#Model Training and Evaluation

We tried 3 models for this dataset. We trained Random Forest (rf) and Gradient Boosting Machine (gbm) models, because usually these two models have good predictions. We also fitted a Linear Discriminant Analysis (LDA), because it is fast and we wanted to compare its results with other models. In each case, we used parallel processing and trained our models on training dataset and evaluated them on testing dataset to get a feel of out of bag error and improve our models. Our final evaluation for our best model will be on validation dataset.

##Random Forest Model

We used "doParallel" library for parallel processing and since our computer had 8 cores, we used 7 cores to boost our training. We also set the method to "cv" with 10 folds cross validation for parameter tuning (with length equal to 10) and metric approximations.


```r
#first prediction -  RF
#starting doParallel
library(doParallel)
set.seed(1)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
tc <- trainControl(method = "cv", number = 10)
modelFitRf <- train(classe ~ ., data = trainingExCor, method = "rf", trControl = tc, tuneLength = 10)
stopCluster(cl)
modelFitRf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 12
## 
##         OOB estimate of  error rate: 0.71%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3338    6    1    1    2 0.002986858
## B   10 2260    9    0    0 0.008336990
## C    0   22 2027    5    0 0.013145083
## D    0    0   19 1911    0 0.009844560
## E    0    0    2    7 2156 0.004157044
```

```r
#modelFitRf
predictTrainRf <- predict(modelFitRf, newdata = trainingExCor)
confusionMatrix(predictTrainRf, trainingExCor$classe)   # 1 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictTestRf <- predict(modelFitRf, newdata = testingExCor)
confusionMatrix(predictTestRf, testingExCor$classe)     # 0.9924
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    8    0    0    0
##          B    2  745    6    0    0
##          C    0    6  676    4    0
##          D    0    0    2  639    2
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9891, 0.9948)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9816   0.9883   0.9938   0.9972
## Specificity            0.9971   0.9975   0.9969   0.9988   1.0000
## Pos Pred Value         0.9929   0.9894   0.9854   0.9938   1.0000
## Neg Pred Value         0.9993   0.9956   0.9975   0.9988   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1899   0.1723   0.1629   0.1833
## Detection Prevalence   0.2860   0.1919   0.1749   0.1639   0.1833
## Balanced Accuracy      0.9977   0.9895   0.9926   0.9963   0.9986
```

By looking at confusion matrix, we can see that our Random Forest model had accuracy of 1 on training data and 0.9924 on testing data. OOB estimate of error rate using 10-fold CV is: 0.71%

##Gradient Boosting Machine Model

We used "doParallel" library and 10 fold cross validation again.


```r
#second prediction - gbm
set.seed(1)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
tc <- trainControl(method = "cv", number = 10)
modelFitGbm <- train(classe ~ ., data = trainingExCor, method = "gbm", trControl = tc, tuneLength = 10, verbose = FALSE)
stopCluster(cl)
modelFitGbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 500 iterations were performed.
## There were 47 predictors of which 47 had non-zero influence.
```

```r
#modelFitGbm
predictTrainGbm <- predict(modelFitGbm, newdata = trainingExCor)
confusionMatrix(predictTrainGbm, trainingExCor$classe)  # 1 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictTestGbm <- predict(modelFitGbm, newdata = testingExCor)
confusionMatrix(predictTestGbm, testingExCor$classe)    # 0.9936 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    4    0    0    0
##          B    1  750    5    0    1
##          C    0    5  678    5    0
##          D    1    0    1  638    2
##          E    0    0    0    0  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9936          
##                  95% CI : (0.9906, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9919          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9881   0.9912   0.9922   0.9958
## Specificity            0.9986   0.9978   0.9969   0.9988   1.0000
## Pos Pred Value         0.9964   0.9908   0.9855   0.9938   1.0000
## Neg Pred Value         0.9993   0.9972   0.9981   0.9985   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1912   0.1728   0.1626   0.1830
## Detection Prevalence   0.2850   0.1930   0.1754   0.1637   0.1830
## Balanced Accuracy      0.9984   0.9930   0.9941   0.9955   0.9979
```

By looking at confusion matrix, we can see that our Gradient Boosting Machine model had accuracy of 1 on training data and 0.9936 on testing data. So "gbm" has a slightly better performance on testing dataset (compared to "rf").

##Linear Discriminant Analysis Model

We used "doParallel" library and 10 fold cross validation again.


```r
#third prediction - lda
set.seed(1)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
tc <- trainControl(method = "cv", number = 10)
modelFitLda <- train(classe ~ ., data = trainingExCor, method = "lda", trControl = tc, tuneLength = 10)
stopCluster(cl)
#modelFitLda$finalModel
modelFitLda
```

```
## Linear Discriminant Analysis 
## 
## 11776 samples
##    47 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10599, 10598, 10599, 10599, 10597, 10599, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD 
##   0.6865677  0.6029901  0.01338224   0.0167681
## 
## 
```

```r
predictTrainLda <- predict(modelFitLda, newdata = trainingExCor)
confusionMatrix(predictTrainLda, trainingExCor$classe)  # 0.6912 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2759  339  230  119   99
##          B   92 1470  193  146  374
##          C  262  272 1330  241  192
##          D  219   91  259 1313  232
##          E   16  107   42  111 1268
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6912          
##                  95% CI : (0.6828, 0.6996)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6089          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8241   0.6450   0.6475   0.6803   0.5857
## Specificity            0.9066   0.9152   0.9005   0.9186   0.9713
## Pos Pred Value         0.7781   0.6462   0.5790   0.6211   0.8212
## Neg Pred Value         0.9284   0.9149   0.9236   0.9361   0.9123
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2343   0.1248   0.1129   0.1115   0.1077
## Detection Prevalence   0.3011   0.1932   0.1951   0.1795   0.1311
## Balanced Accuracy      0.8653   0.7801   0.7740   0.7995   0.7785
```

```r
predictTestLda <- predict(modelFitLda, newdata = testingExCor)
confusionMatrix(predictTestLda, testingExCor$classe)    # 0.6791 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 920 139  82  41  38
##          B  40 464  73  40 111
##          C  72  86 443  87  79
##          D  80  33  73 426  82
##          E   4  37  13  49 411
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6791          
##                  95% CI : (0.6642, 0.6937)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.593           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8244   0.6113   0.6477   0.6625   0.5700
## Specificity            0.8931   0.9166   0.9000   0.9183   0.9678
## Pos Pred Value         0.7541   0.6374   0.5776   0.6138   0.7996
## Neg Pred Value         0.9275   0.9077   0.9236   0.9328   0.9091
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2345   0.1183   0.1129   0.1086   0.1048
## Detection Prevalence   0.3110   0.1856   0.1955   0.1769   0.1310
## Balanced Accuracy      0.8587   0.7639   0.7738   0.7904   0.7689
```

By looking at confusion matrix, we can see that our Linear Discriminant Analysis model had accuracy of 0.6912 on training data and 0.6791 on testing data. OOB estimate of accuracy for 10-fold CV is: 0.6865677. So its performance is much worse than "rf" and "gbm" and is not even comparable!

#Model Selection - Stacked Model

If we were to select just one of these models, we would select "gbm", as it had the largest accuracy. But we would go a step further and make a stacking from these three models. For this purpose, we use the predictions on testing set and create a data frame with each predictions from each model and the true values of "classe" for testing set. Then we will use a linear SVM to stack these values.


```r
#stacking with svm
predTestDf <- data.frame(predRf = predictTestRf, predGbm = predictTestGbm, predLda = predictTestLda, classe = testingExCor$classe)
set.seed(1)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
#tc <- trainControl(method = "cv", number = 10)
combModFit <- train(classe ~ ., data = predTestDf, method = "svmLinear")
stopCluster(cl)
predictTestComb <- predict(combModFit, newdata = predTestDf)
confusionMatrix(predictTestComb, testingExCor$classe)   # 0.9949 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  751    5    0    0
##          C    0    5  676    3    0
##          D    0    0    3  640    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9921, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9883   0.9953   0.9986
## Specificity            0.9989   0.9984   0.9975   0.9988   1.0000
## Pos Pred Value         0.9973   0.9934   0.9883   0.9938   1.0000
## Neg Pred Value         1.0000   0.9975   0.9975   0.9991   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1914   0.1723   0.1631   0.1835
## Detection Prevalence   0.2852   0.1927   0.1744   0.1642   0.1835
## Balanced Accuracy      0.9995   0.9939   0.9929   0.9971   0.9993
```

As we can see, this stacking model has higher accuracy than any of three models used in it, with 0.9949 accuracy on testing set. But this metric is somehaw optimistic, because we used testing set for training this stacking. So lets check it on the validation set to get an estimate of out of bag accuracy and error.

#Final Model Evaluation

We will use the stacking model from last section for prediction, so we want to know its out of bag error. We will predict validation set for this purpose:


```r
#validation
#building validation predictions data frame
predictValidationRf <- predict(modelFitRf, newdata = validationExCor)
predictValidationGbm <- predict(modelFitGbm, newdata = validationExCor)
predictValidationLda <- predict(modelFitLda, newdata = validationExCor)
#predicting on validation set
predValidationDf <- data.frame(predRf = predictValidationRf, predGbm = predictValidationGbm, predLda = predictValidationLda, classe = validationExCor$classe)
predictValidationComb <- predict(combModFit, predValidationDf)
confusionMatrix(predictValidationComb, validationExCor$classe)  #0.9941 accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  754    5    0    0
##          C    0    2  675    5    0
##          D    0    0    4  634    0
##          E    0    0    0    4  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9912, 0.9963)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9926          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9934   0.9868   0.9860   1.0000
## Specificity            0.9989   0.9984   0.9978   0.9988   0.9988
## Pos Pred Value         0.9973   0.9934   0.9897   0.9937   0.9945
## Neg Pred Value         1.0000   0.9984   0.9972   0.9973   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1922   0.1721   0.1616   0.1838
## Detection Prevalence   0.2852   0.1935   0.1738   0.1626   0.1848
## Balanced Accuracy      0.9995   0.9959   0.9923   0.9924   0.9994
```

So we expect our stacked model to have an accuracy of 0.9941 or out of bag error of 0.0059.

#Final Test Set Classification

We have to predict "classe" for 20 final set observation and write them to some files. We will use a funtion for writing these values to files:


```r
#write to text function
pml_write_files = function(x, folder){
    n = length(x)
    for(i in 1:n){
        filename = paste0("./", folder, "/","problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
```

To make prediction on final test set, we should do the same things that we did for our trainig dataset.


```r
#final test
finalTest <- read.csv("./data/pml-testing.csv")
finalTestUsefull <- finalTest[, -c(1:7)]
finalTestFiltered <- finalTestUsefull[, -nzv]
finalTestEx <- finalTestFiltered[, -excludeCols]
finalTestExCor <- finalTestEx[, -corVars]
```

Then we can make our prediction using our stacked model:


```r
#prediction
predictFinalTestRf <- predict(modelFitRf, newdata = finalTestExCor)
predictFinalTestGbm <- predict(modelFitGbm, newdata = finalTestExCor)
predictFinalTestLda <- predict(modelFitLda, newdata = finalTestExCor)
predFinalTestDf <- data.frame(predRf = predictFinalTestRf, predGbm = predictFinalTestGbm, predLda = predictFinalTestLda)
predictFinalTestComb <- predict(combModFit, newdata = predFinalTestDf)
predictFinalTestComb
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Finally we would use the "pml_write_files" function to write these values to files in a "preds" directory:


```r
#write to text files
dir.create("./preds")
```

```
## Warning in dir.create("./preds"): './preds' already exists
```

```r
pml_write_files(predictFinalTestComb, "preds")
```
