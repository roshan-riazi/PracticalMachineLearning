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
FullData <- read.csv("./data/pml-training.csv")
head(FullData)
str(FullData, list.len = 160)
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
#removing unrelated variables
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
validation <- validation[, -c(1:7)]
str(training, list.len = 153)
#changing factors to numeric
for(i in 1:152){
    if(class(training[, i]) == "factor"){
        training[, i] <- as.numeric(as.factor(training[, i]))
        testing[, i] <- as.numeric(as.factor(testing[, i]))
        validation[, i] <- as.numeric(as.factor(validation[, i]))
    }
}
str(FullDataUsefull, list.len = 153)
summary(FullData$classe)
#removing near zero variance variables
nzv <- nearZeroVar(training[, -153])
length(nzv)
trainingFiltered <- training[, -nzv]
testingFiltered <- testing[, -nzv]
validationFiltered <- validation[, -nzv]
dim(trainingFiltered)
#checking for redundant variables
excludeCols <- grep("^max|^min|^amplitude|^var|^avg|^stddev", x = names(trainingFiltered))
trainingEx <- trainingFiltered[, -excludeCols]
testingEx <- testingFiltered[, -excludeCols]
validationEx <- validationFiltered[, -excludeCols]
#removing correlated variables with cutoff = 0.95
corVars <- findCorrelation(x = cor(trainingEx[, -53]), cutoff = 0.95)
trainingExCor <- trainingEx[, -corVars]
testingExCor <- testingEx[, -corVars]
validationExCor <- validationEx[, -corVars]

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

#================================
#stacking with svm
#predTrainDf <- data.frame(predRf = predictTrainRf, predGbm = predictTrainGbm, predLda = predictTrainLda, classe = trainingExCor$classe)
predTestDf <- data.frame(predRf = predictTestRf, predGbm = predictTestGbm, predLda = predictTestLda, classe = testingExCor$classe)

#predTrainDf <- data.frame(predRf = predictTrainRf, predGbm = predictTrainGbm, classe = trainingExCor$classe)
#predTestDf <- data.frame(predRf = predictTestRf, predGbm = predictTestGbm, classe = testingExCor$classe)

set.seed(1)
combModFit <- train(classe ~ ., data = predTestDf, method = "svmLinear")
#predictTrainComb <- predict(combModFit, newdata = predTrainDf)
#confusionMatrix(predictTrainComb, trainingExCor$classe) # 1 accuracy
predictTestComb <- predict(combModFit, newdata = predTestDf)
confusionMatrix(predictTestComb, testingExCor$classe)   # 0.9949 accuracy

#================================
#validation
#gmb performed better on testing data, even better than stacking. So we will just use gbm model!
#confusionMatrix(predictValidationGbm, validationExCor$classe)   # 0.9939 accuracy
#building validation predictions data frame
predictValidationRf <- predict(modelFitRf, newdata = validationExCor)
predictValidationGbm <- predict(modelFitGbm, newdata = validationExCor)
predictValidationLda <- predict(modelFitLda, newdata = validationExCor)
predValidationDf <- data.frame(predRf = predictValidationRf, predGbm = predictValidationGbm, predLda = predictValidationLda, classe = validationExCor$classe)
predictValidationComb <- predict(combModFit, predValidationDf)
confusionMatrix(predictValidationComb, validationExCor$classe)  #0.9941 accuracy

#================================
#write to text function
pml_write_files = function(x, folder){
    n = length(x)
    for(i in 1:n){
        filename = paste0("./", folder, "/","problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

#================================
#final test
finalTest <- read.csv("./data/pml-testing.csv")
finalTestUsefull <- finalTest[, -c(1:7)]
finalTestFiltered <- finalTestUsefull[, -nzv]
finalTestEx <- finalTestFiltered[, -excludeCols]
finalTestExCor <- finalTestEx[, -corVars]
#prediction
predictFinalTestRf <- predict(modelFitRf, newdata = finalTestExCor)
predictFinalTestGbm <- predict(modelFitGbm, newdata = finalTestExCor)
predictFinalTestLda <- predict(modelFitLda, newdata = finalTestExCor)
predFinalTestDf <- data.frame(predRf = predictFinalTestRf, predGbm = predictFinalTestGbm, predLda = predictFinalTestLda)
predictFinalTestComb <- predict(combModFit, newdata = predFinalTestDf)
predictFinalTestComb
#write to text files
dir.create("./preds")
pml_write_files(predictFinalTestComb, "preds")
