

######################################################################################
###  This code allows to obtain similar results to those of Table 2
###  and Figure 2 and Figure 3 in the paper:
###  "Methodological guidance on clinical prediction models for mental health problems"
######################################################################################


# EXAMPLE with NOISY DATA


# Fitting different models on Noisy data

# Models:
# (1) - Linear model via lm()
# (2) - boosting model via glmboost()
# (3) - lasso model via glmnet()
# (4) - random forest model via randomForest()
# (5) - tree-based approach via gradient boosting xgboost()
# (6) - neural network via keras_model_sequential()


library("RColorBrewer")
library(xtable)
library(mboost)
library(glmnet)
library(randomForest)
library(here)
library(rstudioapi)

# automatically set your working directory to the script directory  
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

noisy_datp <- read.table("noisy_PPD_data_18variables.txt", header = T, sep = " ")
colnames(noisy_datp) <- gsub("_noisy$", "", colnames(noisy_datp))

###########==========================================================
# Define the predictors and the response variable
predictors <- c("AgeMother", "born_foreign", "live_partner", "SES1", "SES2",
                "hist_mentprob", "FirstPregnancy", "adv_obstetrics",
                "singmult_preg", "healthcare", "socialsupport", 
                "health_child", "health_partner", "cancer_household", 
                "mental_household")

response <- "EPDS"

########################
# Predictors: 
# 
# AgeMother : age of the mother
# born_foreign: not born in the country of residence? (binary)
# live_partner: living with partner (binary)
# SES1: lower education (binary)
# SES2: unemployed (binary)
# hist_mentprob: history of mental problems (binary)
# FirstPregnancy: is that the first pregnancy (binary)
# adv_obstetrics: any adverse obstetrics (binary)
# singmult_preg: singleton or multiple pregnancy (binary)
# healthcare: supported by primary maternity care provider (ordered 1:3)
# socialsupport: supported by social network (ordered 1:7)
# health_child: any health issues with the child (binary)
# health_partner: any health issues with the partner (binary)
# cancer_household: cancer case in child or partner (binary)
# mental_household: mental illness in child or partner (binary)
# 
#########################################################################




# transform the categorical predictors to factors in the noisy_datp dataset
noisy_datp[3:18] <- lapply(noisy_datp[3:18], function(x) {as.factor(as.vector(x))})


# split by time
noisy_datp.train.time <- noisy_datp[noisy_datp$date ==  0,]
noisy_datp.test.time  <- noisy_datp[noisy_datp$date ==  1,]

# split by Region (country)
noisy_datp.train.reg <- noisy_datp[noisy_datp$country ==  1,]
noisy_datp.test.reg  <- noisy_datp[noisy_datp$country ==  0,]



# ----------- (1)  temporal validation: split by time

# classical linear model
lm1 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
                 SES1 + SES2 + hist_mentprob + FirstPregnancy +
                 adv_obstetrics + singmult_preg + healthcare + 
                 socialsupport + health_child + health_partner + 
                 cancer_household + mental_household , data = noisy_datp.train.time)
summary(lm1)

# compute prediction on test data
preds <- predict(lm1, newdata= noisy_datp.test.time)

# predictive R^2 
cor(preds, noisy_datp.test.time$EPDS, use = "pairwise.complete.obs")^2
# rmsep
sqrt(mean((preds - noisy_datp.test.time$EPDS)^2, na.rm = TRUE))


# 
length(coef(lm1))


# ---------------------
# construct a boosting model
library(mboost)
lm1_boost <- glmboost(EPDS ~ AgeMother + born_foreign  + live_partner + 
                        SES1 + SES2 + hist_mentprob + FirstPregnancy +
                        adv_obstetrics + singmult_preg + healthcare + 
                        socialsupport + health_child + health_partner + 
                        cancer_household + mental_household , data = noisy_datp.train.time)

set.seed(123)
cvr <- cvrisk(lm1_boost, grid = 1:500, folds = cv(model.weights(lm1_boost), type = "subsampling"))
plot(cvr)
mstop(lm1_boost) <- mstop(cvr)

preds_boost <- predict(lm1_boost, newdata= noisy_datp.test.time)
# predictive R^2 
cor(preds_boost, noisy_datp.test.time$EPDS, use = "pairwise.complete.obs")^2
# rmsep
sqrt(mean((preds_boost - noisy_datp.test.time$EPDS)^2, na.rm = TRUE))


#--------- Lasso
library(glmnet)


# Fit a Lasso model using glmnet
# lasso_model <- glmnet(noisy_datp.train.time[, predictors],
#                       noisy_datp.train.time[, response],
#                       alpha = 1,
#                       lambda = 0.0528, nlambda=500)
# plot(lasso_model, xvar = "norm", label = TRUE)  # xvar = "norm" for L1 norm; 
set.seed(2305)

# Define a sequence of lambda values
lambda_seq <- 10^seq(3, -3, length.out = 100)
# Fit the LASSO model using this sequence
lasso_model <- glmnet(noisy_datp.train.time[, predictors],
                      noisy_datp.train.time[, response],
                      alpha = 1,
                      lambda = lambda_seq)
plot(lasso_model, xvar = "norm", label = TRUE)  # xvar = "norm" for L1 norm; Fig3 (left) in the paper


preds_lasso <- predict.glmnet(lasso_model,
                              newx = as.matrix(noisy_datp.test.time[, predictors]))

# predictive R^2 
cor(preds_lasso, noisy_datp.test.time$EPDS, use = "pairwise.complete.obs")^2
# rmsep
sqrt(mean((preds_lasso - noisy_datp.test.time$EPDS)^2, na.rm = TRUE))



## Fit a Lasso with CV  using glmnet

# fit on train.time
set.seed(2305)
cv.lasso <- cv.glmnet(x = as.matrix(noisy_datp.train.time[, predictors]),
                      y = as.matrix(noisy_datp.train.time[, response]))
plot(cv.lasso) # Fig. 3 (right) in the paper 
 
# fit on the entire dataset
cv.lasso_all <- cv.glmnet(x = as.matrix(noisy_datp[, predictors]),
                          y = as.matrix(noisy_datp[, response]))
cv.lasso$lambda.1se

lasso_model_all <- glmnet(noisy_datp[,predictors],
                          as.matrix(noisy_datp[, response]),
                          alpha = 1,
                          lambda = cv.lasso_all$lambda.1se)

round(coef(lasso_model_all),3) # Table 2 in the paper 

## To Latex code
# xtable(as.matrix(coef(lasso_model_all)), digits= 2)



preds_lasso_all <- predict.glmnet(lasso_model_all,
                              s = cv.lasso_all$lambda.1se,
                              newx = as.matrix(noisy_datp.test.time[, predictors]))

# predictive R^2 
cor(preds_lasso_all, noisy_datp.test.time$EPDS, use = "pairwise.complete.obs")^2
# rmsep
sqrt(mean((preds_lasso_all - noisy_datp.test.time$EPDS)^2, na.rm = TRUE))



#######################
library(randomForest)

# Train Random Forest model
set.seed(2305)  # Set seed for reproducibility
rf_model <- randomForest(x = as.matrix(noisy_datp.train.time[, predictors]),
                         y = as.matrix(noisy_datp.train.time[, response]),
                         ntree = 500 )  # Number of trees in the forest

# Predict using the Random Forest model
preds_rf <- predict(rf_model, newdata = as.matrix(noisy_datp.test.time[, predictors]))

# Compute predictive R^2 
r2_rf_reg <- cor(preds_rf, noisy_datp.test.time$EPDS, use = "pairwise.complete.obs")^2
r2_rf_reg
# Compute RMSEP
rmsep_rf_reg <- sqrt(mean((preds_rf - noisy_datp.test.time$EPDS)^2, na.rm = TRUE))
rmsep_rf_reg


#--------- xgboost

library(xgboost)

set.seed(123)

# Convert the training data to a matrix
noisy_datp_numeric_train.reg <- data.frame(lapply(noisy_datp.train.reg, function(x) as.numeric(as.character(x))))
noisy_datp_numeric_test.reg <- data.frame(lapply(noisy_datp.test.reg, function(x) as.numeric(as.character(x))))

dat <- xgb.DMatrix(data = as.matrix(noisy_datp_numeric_train.reg[,predictors]),
                   label = as.matrix(noisy_datp_numeric_train.reg[, response]))


set.seed(2305)
x1_cv <- xgb.cv(data = dat, nfold = 20, nrounds = 100)
which.min(x1_cv$evaluation_log$test_rmse_mean)

x1 <- xgboost(data = dat, nrounds = 11, objective = "reg:squarederror")

# Obtain a plot similar to Fig. 2 in the paper
xgb.ggplot.importance(model = x1, importance_matrix = xgb.importance(model = x1)) 

preds_xgboost <- predict(x1, newdata = as.matrix(noisy_datp_numeric_test.reg[, predictors]))

## predictive R^2 
#r2a(preds_xgboost, datp.test.time$EPDS)
cor(preds_xgboost, noisy_datp_numeric_test.reg$EPDS, use = "pairwise.complete.obs")^2

# rmsep
sqrt(mean((preds_xgboost - noisy_datp_numeric_test.reg$EPDS)^2, na.rm = TRUE))




#---------- keras

library(keras)

Sys.setenv(TENSORFLOW_PYTHON_VERSION = "3.12")
library(tensorflow)

# Set seed for reproducibility
set.seed(123)


# Create a simple neurahl network model 
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", 
              input_shape = ncol(noisy_datp.train.time[, predictors])) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(),
  metrics = c("mean_squared_error")
)


# Convert the training data to a matrix
noisy_datp_numeric_train.time <- data.frame(lapply(noisy_datp.train.time, function(x) as.numeric(as.character(x))))
noisy_datp_numeric_train.time_reduced <- noisy_datp_numeric_train.time[,-((ncol(noisy_datp_numeric_train.time) - 1):ncol(noisy_datp_numeric_train.time))]
noisy_datp_numeric_train.time_reduced <- noisy_datp_numeric_train.time_reduced[,-1] # remove response
noisy_datp_numeric_train.time_reduced <- as.matrix(noisy_datp_numeric_train.time_reduced)
noisy_datp_numeric_test.time <- data.frame(lapply(noisy_datp.test.reg, function(x) as.numeric(as.character(x))))
noisy_datp_numeric_test.time_reduced <- noisy_datp_numeric_test.time[,-((ncol(noisy_datp_numeric_test.time) - 1):ncol(noisy_datp_numeric_test.time))]
noisy_datp_numeric_test.time_reduced  <- noisy_datp_numeric_test.time_reduced[,-1]  # remove response
noisy_datp_numeric_test.time_reduced <- as.matrix(noisy_datp_numeric_test.time_reduced)


train_labels <- noisy_datp_numeric_train.time[, response]
test_labels <- noisy_datp_numeric_test.time[, response]

# Train the model
history <- model %>% fit(
  noisy_datp_numeric_train.time_reduced, train_labels,
  epochs = 50,  # Number of training epochs (equivalent to nrounds in xgboost)
  batch_size = 32,  # Batch size
  validation_split = 0.2  # Use 20% of the data for validation
)



# Make predictions on the test data
preds_keras <- model %>% predict(noisy_datp_numeric_test.time_reduced)

# Compute predictive R^2
cor(preds_keras, noisy_datp_numeric_test.time$EPDS, use = "pairwise.complete.obs")^2

# rmsep
sqrt(mean((preds_keras - noisy_datp_numeric_test.time$EPDS)^2, na.rm = TRUE))





# ------- (2) regional validation: split by Country

# classical linear model
lm2 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
            SES1 + SES2 + hist_mentprob + FirstPregnancy +
            adv_obstetrics + singmult_preg + healthcare + 
            socialsupport + health_child + health_partner + 
            cancer_household + mental_household , data = noisy_datp.train.reg)
summary(lm2)

# compute prediction on test data
preds2 <- predict(lm2, newdata= noisy_datp.test.reg)

# predictive R^2 
cor(preds2, noisy_datp.test.reg$EPDS, use = "pairwise.complete.obs")^2
# rmsep
sqrt(mean((preds2 - noisy_datp.test.reg$EPDS)^2, na.rm = TRUE))


