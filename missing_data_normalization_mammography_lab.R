#-- Install the required packages

library(caret)
library(e1071)
library(ggplot2)
library(dplyr)
install.packages(c("unbalanced",
                    "GGally",
                   "missForest"))

library(unbalanced)
library(GGally)

install.packages(c("parallel",
                   "doMC",
                   "doParallel"))

library(parallel)
library(doMC)
library(doParallel)
install.packages("missForest")
library(missForest)

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

setwd("~/Lab_Session_1/")
dir()

names.mam <- c("BI-RADs",
               "Age",
               "Shape",
               "Margin",
               "Density",
               "Severity")

mammography <- read.table("mammographic_masses.data",
                          header = F,
                          sep = ",",
                          col.names = names.mam,
                          na.strings = "?")

summary(mammography)
View(mammography)
sum(is.na(mammography))

no.na.mammography <- na.omit(mammography)
View(no.na.mammography)

#View(table(no.na.mammography$BI.RADs))

no.na.mammography[no.na.mammography$BI.RADs == 55, 1] <- 5

table(no.na.mammography$BI.RADs)
table(no.na.mammography$Severity)

#View(maxmindf)

no.na.mammography$Severity <- as.factor(no.na.mammography$Severity)
str(no.na.mammography)
View(no.na.mammography)

ggpairs(no.na.mammography,
        title = "Scatterplot Matrix of the Features of the Mammography Severity Dataset")

set.seed(6651)

intrain <- createDataPartition(y = no.na.mammography$Severity,
                               p= 0.7, 
                               list = FALSE)

training <- no.na.mammography[intrain,]
testing <- no.na.mammography[-intrain,]

anyNA(no.na.mammography)
summary(no.na.mammography)
str(no.na.mammography)

library(e1071)
set.seed(1066)
svm.model <- svm(Severity~.,
                 data = training,
                 kernel = "radial",
                 cross = 10,
                 gamma = 0.2,
                 cost = 1,
                 fitted = TRUE,
                 probability = TRUE,
                 type = "C-classification")

train.pred <- predict(svm.model,
        training)

confusionMatrix(train.pred,
                training$Severity)

test.pred <- predict(svm.model,
                      testing)

confusionMatrix(test.pred,
                testing$Severity)

#-------- Missing Data Treatment with RFs

summary(mammography)

set.seed(314)
mammography.imp <- missForest(mammography,
                              ntree = 1500,
                              mtry = 3)

imputed.mammography.df <- mammography.imp$ximp

View(imputed.mammography.df)
mammography.imp$OOBerror

set.seed(1010)
imp.index <- createDataPartition(imputed.mammography.df$Severity,
                                 p = 0.7,
                                 list = FALSE)

imp.training <- imputed.mammography.df[imp.index,]
imp.testing <- imputed.mammography.df[-imp.index,]

imp.training$Severity <- as.factor(imp.training$Severity)
class(imp.training$Severity)
imp.testing$Severity <- as.factor(imp.testing$Severity)
class(imp.testing$Severity)

set.seed(1010)
imp.svm.model <- svm(Severity~.,
                 data = imp.training,
                 kernel = "radial",
                 cross = 10,
                 gamma = 0.2,
                 cost = 1,
                 fitted = TRUE,
                 probability = TRUE,
                 type = "C-classification")

imp.train.pred <- predict(imp.svm.model,
                          imp.training)

confusionMatrix(imp.train.pred,
                imp.training$Severity)

imp.test.pred <- predict(imp.svm.model,
                     imp.testing,)

#attr(imp.test.pred, "probabilities")[1:5,]

confusionMatrix(imp.test.pred,
                imp.testing$Severity)

#--------- Normalización

# we assime that NAs are ommited
no.na.mammography
# log2 normalization
Label.Severity <- no.na.mammography$Severity
log2.no.na.mammography <- log2(no.na.mammography[,1:5])
log2.no.na.mammography <- cbind(log2.no.na.mammography,
                                Label.Severity)

Severity <- no.na.mammography$Severity
scaleddata<-scale(no.na.mammography[,1:5],
                  center = TRUE,
                  scale = TRUE)
scaleddata <- cbind(scaleddata,
                    Severity)
scaleddata <- as.data.frame(scaleddata)


# Min-max normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
maxmindf <- as.data.frame(lapply(no.na.mammography[,1:5], 
                                normalize))

maxmindf <- cbind(maxmindf,
                  Severity)

set.seed(1111)

norm.index <- createDataPartition(scaleddata$Severity,
                                  p = 0.7,
                                  list = FALSE)

norm.train = scaleddata[norm.index,]
norm.test = scaleddata[-norm.index,]

norm.train$Severity <- as.factor( norm.train$Severity)
norm.test$Severity <- as.factor( norm.test$Severity)
set.seed(1010)
norm.svm.model <- svm(Severity~.,
                     data = norm.train,
                     kernel = "radial",
                     cross = 10,
                     gamma = 0.2,
                     cost = 1,
                     fitted = TRUE,
                     probability = TRUE,
                     type = "C-classification")


norm.train.pred <- predict(norm.svm.model,
                          norm.train)

confusionMatrix(norm.train.pred,
                norm.train$Severity)

norm.test.pred <- predict(norm.svm.model,
                         norm.test,)

confusionMatrix(norm.test.pred,
                norm.test$Severity)
