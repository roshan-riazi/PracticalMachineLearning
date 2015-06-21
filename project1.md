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
modelFitRf
predictTrainRf <- predict(modelFitRf, newdata = trainingExCor)
confusionMatrix(predictTrainRf, trainingExCor$classe)   # 1 accuracy
predictTestRf <- predict(modelFitRf, newdata = testingExCor)
confusionMatrix(predictTestRf, testingExCor$classe)     # 0.9924
```

By looking at confusion matrix, we can see that our Random Forest model had accuracy of 1 on training data and 0.9924 on testing data.

##Gradient Boosting Machine Model

We used "doParallel" library and 10 fold cross validation again.


```r
#second prediction - gbm
set.seed(1)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
tc <- trainControl(method = "cv", number = 10)
modelFitGbm <- train(classe ~ ., data = trainingExCor, method = "gbm", trControl = tc, tuneLength = 10)
stopCluster(cl)
modelFitGbm$finalModel
modelFitGbm
predictTrainGbm <- predict(modelFitGbm, newdata = trainingExCor)
confusionMatrix(predictTrainGbm, trainingExCor$classe)  # 1 accuracy
predictTestGbm <- predict(modelFitGbm, newdata = testingExCor)
confusionMatrix(predictTestGbm, testingExCor$classe)    # 0.9936 accuracy
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
modelFitLda$finalModel
modelFitLda
predictTrainLda <- predict(modelFitLda, newdata = trainingExCor)
confusionMatrix(predictTrainLda, trainingExCor$classe)  # 0.6912 accuracy
predictTestLda <- predict(modelFitLda, newdata = testingExCor)
confusionMatrix(predictTestLda, testingExCor$classe)    # 0.6791 accuracy
```

By looking at confusion matrix, we can see that our Linear Discriminant Analysis model had accuracy of 0.6912 on training data and 0.6791 on testing data. So its performance is much worse than "rf" and "gbm" and is not even comparable!

#Model Selection - Stacked Model

If we were to select just one of these models, we would select "gbm", as it had the largest accuracy. But we would go a step further and make a stacking from these three models. For this purpose, we use the predictions on testing set and create a data frame with each predictions from each model and the true values of "classe" for testing set. Then we will use a linear SVM to stack these values.


```r
#stacking with svm
predTestDf <- data.frame(predRf = predictTestRf, predGbm = predictTestGbm, predLda = predictTestLda, classe = testingExCor$classe)
set.seed(1)
combModFit <- train(classe ~ ., data = predTestDf, method = "svmLinear")
predictTestComb <- predict(combModFit, newdata = predTestDf)
confusionMatrix(predictTestComb, testingExCor$classe)   # 0.9949 accuracy
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

So we expect our stacked model to have and accuracy of 0.9941 or out of bag error of 0.0059.

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
predictFinalTestComb <- predict(combModFit, newdata = finalTestExCor)
predictFinalTestComb
```

Finally we would use the "pml_write_files" function to write these values to files in a "preds" directory:


```r
#write to text files
dir.create("./preds")
pml_write_files(predictFinalTestGbm, "preds")
```
