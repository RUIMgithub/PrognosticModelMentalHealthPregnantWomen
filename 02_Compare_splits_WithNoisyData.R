
######################################################################################
###  This code allows to obtain similar results to those of Table 3 in the paper:
###  "Methodological guidance on clinical prediction models for mental health problems"
######################################################################################

# EXAMPLE with NOISY DATA

# Comparing splits on Noisy data using:

# (1) split by time
# (2) split by country
# (3) random split 
# (4) cv / bootstrapping




datp <- read.table("noisy_PPD_data_18variables.txt", header = T, sep = " ")
colnames(datp) <- gsub("_noisy$", "", colnames(datp))


library("RColorBrewer")
library(xtable)
library(mboost)
library(glmnet)
library(randomForest)


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
datp[3:18] <- lapply(datp[3:18], function(x) {as.factor(as.vector(x))})


##----------------------
## Splits

# split by time
datp.train.time <- datp[datp$date ==  0,]
datp.test.time  <- datp[datp$date ==  1,]

# split by Region (country)
datp.train.reg <- datp[datp$country ==  1,]
datp.test.reg  <- datp[datp$country ==  0,]

# generate random split (2/3 of the initial dataset for training and 1/3 for testing)
set.seed(2305)
insample <- sample(1:nrow(datp), round((2/3)*nrow(datp)))
datp.train.ran <- datp[(1:nrow(datp)) %in% insample,] 
datp.test.ran <- datp[!(1:nrow(datp)) %in% insample,] 



# (1) split by time
# (2) split by country
# (3) random split 
# (4) cv / bootstrapping


# Function to calculate Coefficient of determination
r2a <- function(preds, test){
  
  preds <- preds[!is.na(preds)&!is.na(test)]
  test  <- test[!is.na(preds)&!is.na(test)]
  mean_test <- mean(test)
  
  1 - (sum((preds-test)^2)/sum((test-mean_test)^2))
}



# ----------- (1)  temporal validation: split by time


#-------------  classical linear model
lm1 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
            SES1 + SES2 + hist_mentprob + FirstPregnancy +
            adv_obstetrics + singmult_preg + healthcare + 
            socialsupport + health_child + health_partner + 
            cancer_household + mental_household , data = datp.train.time)
summary(lm1)
lm1 <- step(lm1, direction = "backward")

# compute prediction on test data
preds <- predict(lm1, newdata= datp.test.time)

# predictive R^2 
#r2_lm_time <- cor(preds, datp.test.time$EPDS, use = "pairwise.complete.obs")^2
r2_lm_time <- r2a(preds = preds, test = datp.test.time$EPDS)

# rmsep
rmsep_lm_time <- sqrt(mean((preds - datp.test.time$EPDS)^2, na.rm = TRUE))
rmsep_lm_time  # 

#------------- boosting
lm1_boost <- glmboost(EPDS ~ AgeMother + born_foreign  + live_partner + 
                        SES1 + SES2 + hist_mentprob + FirstPregnancy +
                        adv_obstetrics + singmult_preg + healthcare + 
                        socialsupport + health_child + health_partner + 
                        cancer_household + mental_household , data = datp.train.time)

set.seed(2305)
cvr <- cvrisk(lm1_boost, grid = 1:300, folds = cv(model.weights(lm1_boost), type = "bootstrap"))
mstop(lm1_boost) <- mstop(cvr)

preds_boost <- predict(lm1_boost, newdata= datp.test.time)
# predictive R^2 
r2_boost_time <- r2a(preds_boost, test = datp.test.time$EPDS)
# rmsep
rmsep_boost_time <- sqrt(mean((preds_boost - datp.test.time$EPDS)^2, na.rm = TRUE))
rmsep_boost_time #


#---------------
#---------------  Lasso model using glmnet
lasso_model <- glmnet(as.matrix(datp.train.time[, predictors]), as.matrix(datp.train.time[, response]), alpha = 1)

set.seed(2305)
cv.lasso <- cv.glmnet(x = as.matrix(datp.train.time[, predictors]), y = as.matrix(datp.train.time[, response]))

preds_lasso <- predict.glmnet(lasso_model, s = cv.lasso$lambda.1se, newx = as.matrix(datp.test.time[, predictors]))

# predictive R^2 
r2_lasso_time <- r2a(preds = preds_lasso, test = datp.test.time$EPDS)
# rmsep
rmsep_lasso_time <- sqrt(mean((preds_lasso - datp.test.time$EPDS)^2, na.rm = TRUE))
rmsep_lasso_time # 

#--------------
#-------------- Random Forest model
set.seed(2305)  # Set seed for reproducibility
rf_model <- randomForest(x = datp.train.time[, predictors],
                         y = datp.train.time[, response],
                         ntree = 1000, mtry = 10 )  # Number of trees in the forest

# Predict using the Random Forest model
preds_rf <- predict(rf_model, newdata = datp.test.time[, predictors])

# Compute predictive R^2 
r2_rf_time <- r2a(preds_rf, test = datp.test.time$EPDS)

# Compute RMSEP
rmsep_rf_time <- sqrt(mean((preds_rf - datp.test.time$EPDS)^2, na.rm = TRUE))
rmsep_rf_time # 6.158058

#-----------------------------------------------------



# ----------- (2)  regional validation, split by Region

#-------------  classical linear model
lm1 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
            SES1 + SES2 + hist_mentprob + FirstPregnancy +
            adv_obstetrics + singmult_preg + healthcare + 
            socialsupport + health_child + health_partner + 
            cancer_household + mental_household , data = datp.train.reg)
summary(lm1)
lm1 <- step(lm1, direction = "backward")

# compute prediction on test data
preds <- predict(lm1, newdata= datp.test.reg)

# predictive R^2 
r2_lm_reg <- r2a(preds, test =  datp.test.reg$EPDS)
# rmsep
rmsep_lm_reg <- sqrt(mean((preds - datp.test.reg$EPDS)^2, na.rm = TRUE))
rmsep_lm_reg # 


#-------------  
#------------- boosting

lm1_boost <- glmboost(EPDS ~ AgeMother + born_foreign  + live_partner + 
                        SES1 + SES2 + hist_mentprob + FirstPregnancy +
                        adv_obstetrics + singmult_preg + healthcare + 
                        socialsupport + health_child + health_partner + 
                        cancer_household + mental_household , data = datp.train.reg)

set.seed(2305)
cvr <- cvrisk(lm1_boost, grid = 1:300, folds = cv(model.weights(lm1_boost), type = "bootstrap"))
mstop(lm1_boost) <- mstop(cvr)

preds_boost <- predict(lm1_boost, newdata= datp.test.reg)
# predictive R^2 
r2_boost_reg <- r2a(preds_boost, test =  datp.test.reg$EPDS)
# rmsep
rmsep_boost_reg <- sqrt(mean((preds_boost - datp.test.reg$EPDS)^2, na.rm = TRUE))
rmsep_boost_reg # 
#---------------


#---------------  Lasso model using glmnet
lasso_model <- glmnet(as.matrix(datp.train.reg[, predictors]), as.matrix(datp.train.reg[, response]), alpha = 1)

set.seed(2305)
cv.lasso <- cv.glmnet(x = as.matrix(datp.train.reg[, predictors]), y = as.matrix(datp.train.reg[, response]))

preds_lasso <- predict.glmnet(lasso_model, s = cv.lasso$lambda.1se, newx = as.matrix(datp.test.reg[, predictors]))

# predictive R^2 
r2_lasso_reg <- r2a(preds_lasso, datp.test.reg$EPDS)
# rmsep
rmsep_lasso_reg <- sqrt(mean((preds_lasso - datp.test.reg$EPDS)^2, na.rm = TRUE))
rmsep_lasso_reg # 
#--------------


#-------------- Random Forest model
set.seed(2305)  # Set seed for reproducibility
rf_model <- randomForest(x = datp.train.reg[, predictors],
                         y = datp.train.reg[, response],
                         ntree = 500 )  # Number of trees in the forest

# Predict using the Random Forest model
preds_rf <- predict(rf_model, newdata = datp.test.reg[, predictors])

# Compute predictive R^2 
r2_rf_reg <- cor(preds_rf, datp.test.reg$EPDS, use = "pairwise.complete.obs")^2

# Compute RMSEP
rmsep_rf_reg <- sqrt(mean((preds_rf - datp.test.reg$EPDS)^2, na.rm = TRUE))
rmsep_rf_reg # 6.313698




# ----------- (3)  random split (single)


#-------------  classical linear model
lm1 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
            SES1 + SES2 + hist_mentprob + FirstPregnancy +
            adv_obstetrics + singmult_preg + healthcare + 
            socialsupport + health_child + health_partner + 
            cancer_household + mental_household , data = datp.train.ran)
summary(lm1)
lm1 <- step(lm1, direction = "backward")

# compute prediction on test data
preds <- predict(lm1, newdata= datp.test.ran)

# predictive R^2 
r2_lm_ran <- r2a(preds, datp.test.ran$EPDS)
# rmsep
rmsep_lm_ran <- sqrt(mean((preds - datp.test.ran$EPDS)^2, na.rm = TRUE))
rmsep_lm_ran # 5.849046



#-------------
#------------- boosting

lm1_boost <- glmboost(EPDS ~ AgeMother + born_foreign  + live_partner + 
                        SES1 + SES2 + hist_mentprob + FirstPregnancy +
                        adv_obstetrics + singmult_preg + healthcare + 
                        socialsupport + health_child + health_partner + 
                        cancer_household + mental_household , data = datp.train.ran)

set.seed(2305)
cvr <- cvrisk(lm1_boost, grid = 1:300, folds = cv(model.weights(lm1_boost), type = "bootstrap"))
mstop(lm1_boost) <- mstop(cvr)

preds_boost <- predict(lm1_boost, newdata= datp.test.ran)
# predictive R^2 
r2_boost_ran <- r2a(preds_boost, test = datp.test.ran$EPDS)
# rmsep
rmsep_boost_ran <- sqrt(mean((preds_boost - datp.test.ran$EPDS)^2, na.rm = TRUE))
rmsep_boost_ran # 5.840363



#---------------
#---------------  Lasso model using glmnet
lasso_model <- glmnet(as.matrix(datp.train.ran[, predictors]), as.matrix(datp.train.ran[, response]), alpha = 1)

set.seed(2305)
cv.lasso <- cv.glmnet(x = as.matrix(datp.train.ran[, predictors]), y = as.matrix(datp.train.ran[, response]))

preds_lasso <- predict.glmnet(lasso_model, s = cv.lasso$lambda.1se, newx = as.matrix(datp.test.ran[, predictors]))

# predictive R^2 
r2_lasso_ran <- r2a(preds_lasso, test =  datp.test.ran$EPDS)
# rmsep
rmsep_lasso_ran <- sqrt(mean((preds_lasso - datp.test.ran$EPDS)^2, na.rm = TRUE))
rmsep_lasso_ran # 5.897114



#--------------
#-------------- Random Forest model
set.seed(2305)  # Set seed for reproducibility
rf_model <- randomForest(x = datp.train.ran[, predictors ],
                         y = datp.train.ran[, response],
                         ntree = 1000, mtry = 5)  # Number of trees in the forest

# Predict using the Random Forest model
preds_rf <- predict(rf_model, newdata = datp.test.ran[, predictors])

# Compute predictive R^2 
r2_rf_ran <- r2a(preds_rf, datp.test.ran$EPDS)

# Compute RMSEP
rmsep_rf_ran <- sqrt(mean((preds_rf - datp.test.ran$EPDS)^2, na.rm = TRUE))
rmsep_rf_ran # 5.9401




# ----------- (4)  bootstrapping, cv

set.seed(2305)
cvs <- cv(rep(1, nrow(datp)), type = "kfold")


#-------------- lm
rmsep_lm_cv <- r2_lm_cv <- numeric(ncol(cvs))

for(i in 1:ncol(cvs)){
  datp.train.cv <- datp[cvs[,i] == 1,] 
  datp.test.cv  <- datp[cvs[,i] == 0,] 
  
  lm1 <- lm(EPDS ~ AgeMother + born_foreign  + live_partner + 
              SES1 + SES2 + hist_mentprob + FirstPregnancy +
              adv_obstetrics + singmult_preg + healthcare + 
              socialsupport + health_child + health_partner + 
              cancer_household + mental_household , data = datp.train.cv)
  lm1 <- step(lm1, direction = "backward", trace = FALSE)
  
  # compute prediction on test data
  preds <- predict(lm1, newdata= datp.test.cv)
  
  # predictive R^2 
  r2_lm_cv[i] <- r2a(preds, test= datp.test.cv$EPDS)
  # rmsep
  rmsep_lm_cv[i] <- sqrt(mean((preds - datp.test.cv$EPDS)^2, na.rm = TRUE))
}
rmsep_lm_cv # 5.898591 5.737550 6.296480 6.302996 5.362330 6.115206 6.017279 5.826857 5.486873 5.885112


#-------------- boosting
rmsep_boost_cv <- r2_boost_cv <- numeric(ncol(cvs))

for(i in 1:ncol(cvs)){
  datp.train.cv <- datp[cvs[,i] == 1,] 
  datp.test.cv  <- datp[cvs[,i] == 0,] 
  
  lm1_boost <- glmboost(EPDS ~ AgeMother + born_foreign  + live_partner + 
                          SES1 + SES2 + hist_mentprob + FirstPregnancy +
                          adv_obstetrics + singmult_preg + healthcare + 
                          socialsupport + health_child + health_partner + 
                          cancer_household + mental_household , data = datp.train.cv)
  set.seed(2305)
  cvr <- cvrisk(lm1_boost, grid = 1:300, folds = cv(model.weights(lm1_boost), type = "bootstrap"))
  mstop(lm1_boost) <- mstop(cvr)
  
  preds_boost <- predict(lm1_boost, newdata= datp.test.cv)
  # predictive R^2 
  r2_boost_cv[i] <- r2a(preds_boost, datp.test.cv$EPDS)
  # rmsep
  rmsep_boost_cv[i] <- sqrt(mean((preds_boost - datp.test.cv$EPDS)^2, na.rm = TRUE))
}
rmsep_boost_cv # 5.922482 5.751814 6.313291 6.285787 5.365009 6.125051 5.996386 5.828939 5.480883 5.893976



#-------------- lasso
rmsep_lasso_cv <- r2_lasso_cv <- numeric(ncol(cvs))

for(i in 1:ncol(cvs)){
  datp.train.cv <- datp[cvs[,i] == 1,] 
  datp.test.cv  <- datp[cvs[,i] == 0,] 
  
  lasso_model <- glmnet(as.matrix(datp.train.cv[, predictors]), as.matrix(datp.train.cv[, response]), alpha = 1)
  
  set.seed(2305)
  cv.lasso <- cv.glmnet(x = as.matrix(datp.train.cv[, predictors]), y = as.matrix(datp.train.cv[, response]))
  
  preds_lasso <- predict.glmnet(lasso_model, s = cv.lasso$lambda.1se, newx = as.matrix(datp.test.cv[, predictors]))
  
  # predictive R^2 
  r2_lasso_cv[i] <- r2a(preds_lasso, datp.test.cv$EPDS)
  # rmsep
  rmsep_lasso_cv[i] <- sqrt(mean((preds_lasso - datp.test.cv$EPDS)^2, na.rm = TRUE))
}
rmsep_lasso_cv # 6.000578 5.891476 6.354999 6.309134 5.440220 6.232267 5.951973 5.938132 5.561355 5.929587



#-------------- random forests
rmsep_rf_cv <- r2_rf_cv <- numeric(ncol(cvs))

for(i in 1:ncol(cvs)){
  datp.train.cv <- datp[cvs[,i] == 1,] 
  datp.test.cv  <- datp[cvs[,i] == 0,] 
  
  rf_model <- randomForest(x = datp.train.cv[, predictors],
                           y = datp.train.cv[, response],
                           ntree = 500, )  # Number of trees in the forest
  
  # Predict using the cvdom Forest model
  preds_rf <- predict(rf_model, newdata = datp.test.cv[, predictors])
  
  # Compute predictive R^2 
  r2_rf_cv[i] <- r2a(preds_rf, datp.test.cv$EPDS)
  
  # Compute RMSEP
  rmsep_rf_cv[i] <- sqrt(mean((preds_rf - datp.test.cv$EPDS)^2, na.rm = TRUE))
}
rmsep_rf_cv # 6.024664 5.910688 6.426877 6.322100 5.586423 6.189183 6.176315 5.903082 5.583314 5.949489



#---------------------------------------------------------
#---------------------------------------------------------
# Getting the joint results above  

mean(rmsep_lm_cv) # 5.892927
mean(rmsep_boost_cv) # 5.896362
mean(rmsep_lasso_cv) # 5.960972
mean(rmsep_rf_cv) # 6.007214 

mean(r2_lm_cv) # 0.1513104
mean(r2_boost_cv) # 0.1504149
mean(r2_lasso_cv) # 0.131846
mean(r2_rf_cv) # 0.1177942

round(c("step-wise lm:" = rmsep_lm_time,
        "boosting:" = rmsep_boost_time, 
        "lasso:"  = rmsep_lasso_time,
        "random forest:" = rmsep_rf_time), 3)
# step-wise lm:      boosting:         lasso: random forest: 
#  5.952          5.953          6.061          6.158 


rmsep_lm_time # 5.951998
rmsep_boost_time # 5.952691
rmsep_lasso_time # 6.061241
rmsep_rf_time # 6.158058

rmsep_lm_reg # 6.236221
rmsep_boost_reg # 6.234899
rmsep_lasso_reg # 6.290051
rmsep_rf_reg # 6.313698

# Tables
tab <- c(rmsep_lm_ran, rmsep_boost_ran, rmsep_lasso_ran, rmsep_rf_ran) 
tab <- rbind(tab, c(r2_lm_ran, r2_boost_ran, r2_lasso_ran, r2_rf_ran))
tab <- round(tab,3)
colnames(tab) <- c("step-wise", "boosting", "lasso", "random forest")
rownames(tab) <- c("RMSEP", "R^2")
library(xtable)
xtable(tab, digits = 3)

# Tables
tab <- c(mean(rmsep_lm_cv), mean(rmsep_boost_cv), mean(rmsep_lasso_cv), mean(rmsep_rf_cv)) 
tab <- rbind(tab, c(mean(r2_lm_cv), mean(r2_boost_cv), mean(r2_lasso_cv), mean(r2_rf_cv)))
tab <- round(tab,3)
colnames(tab) <- c("step-wise", "boosting", "lasso", "random forest")
rownames(tab) <- c("RMSEP", "R^2")
xtable(tab, digits = 3)

# Tables
tab <- c(rmsep_lm_time, rmsep_boost_time, rmsep_lasso_time, rmsep_rf_time) 
tab <- rbind(tab, c(r2_lm_time, r2_boost_time, r2_lasso_time, r2_rf_time))
tab <- round(tab,3)
colnames(tab) <- c("step-wise", "boosting", "lasso", "random forest")
rownames(tab) <- c("RMSEP", "R^2")
xtable(tab, digits = 3)

# Tables
tab <- c(rmsep_lm_reg, rmsep_boost_reg, rmsep_lasso_reg, rmsep_rf_reg) 
tab <- rbind(tab, c(r2_lm_reg, r2_boost_reg, r2_lasso_reg, r2_rf_reg))
tab <- round(tab,3)
colnames(tab) <- c("step-wise", "boosting", "lasso", "random forest")
rownames(tab) <- c("RMSEP", "R^2")
xtable(tab, digits = 3)




barplot(c(mean(rmsep_lm_cv), mean(rmsep_boost_cv), mean(rmsep_lasso_cv), mean(rmsep_rf_cv)))

        library(ggplot2)
        
        # Create a data frame with model names and their mean RMSEP values
        models <- c("step-wise", "boosting", "lasso", "random forest")
        mean_rmsep <- c(mean(rmsep_lm_cv), mean(rmsep_boost_cv), mean(rmsep_lasso_cv), mean(rmsep_rf_cv))
        data <- data.frame(models, mean_rmsep)
        
        # Create the bar plot using ggplot2
        ggplot(data, aes(x = models, y = mean_rmsep, fill = models)) +
          geom_bar(stat = "identity") +
          labs(x = "Models", y = "Mean RMSEP") +
          ggtitle("RMSEP for 10-fold CV") +
          theme_minimal() +
          scale_fill_brewer()+
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
       
         # Define colors using ColorBrewer palette
        colors <- brewer.pal(4, "Set3")
        
        # Create a vector of mean RMSEP values
        mean_rmsep <- c(mean(rmsep_lm_cv), mean(rmsep_boost_cv), mean(rmsep_lasso_cv), mean(rmsep_rf_cv))
        
        boxplot(cbind(rmsep_lm_cv, rmsep_boost_cv, rmsep_lasso_cv, rmsep_rf_cv))
        
        # Barplot
        barplot(mean_rmsep, 
                names.arg = c("lm", "boost", "lasso", "rf"),
                col = colors,
                xlab = "Models",
                ylab = "Mean RMSEP",
                main = "Mean RMSEP for Different Models")
        
        
        
        
  