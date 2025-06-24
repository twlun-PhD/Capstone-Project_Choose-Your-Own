## ----setup, include=FALSE-----------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tidyr)
library(dplyr)
library(ggplot2)


## ----dataset------------------------------------------------------------------------------------------
# Load and prepare data
edx <- read.csv("global_ai_tools_students_use.csv", stringsAsFactors = FALSE) %>%
  mutate(
    uses_ai_for_study = factor(uses_ai_for_study, levels = c("False","True"))
  )


## ----Dimensions and structure-------------------------------------------------------------------------
dim(edx)
str(edx)


## ----Descriptive statistics---------------------------------------------------------------------------
#  Demographic distributions
summary(edx$age)
edx %>% count(gender)
edx %>% count(country)
edx %>% count(grade)


## -----------------------------------------------------------------------------------------------------
# Overall AI adoption rate
edx %>% 
  count(uses_ai_for_study) %>% 
  mutate(prop = n / sum(n))

# Frequency of each AI tool
edx %>% 
  select(starts_with("uses_"), -uses_ai_for_study) %>% 
  summarise(across(everything(), ~ sum(. == "True"))) %>% 
  pivot_longer(everything(), names_to = "tool", values_to = "count")


## -----------------------------------------------------------------------------------------------------
# Train-test split (80/20 stratified)
set.seed(1)
trainIndex <- createDataPartition(edx$uses_ai_for_study, p = 0.8, list = FALSE)
train      <- edx[trainIndex, ]
test       <- edx[-trainIndex, ]

# Define your tool names & corresponding usefulness columns
tools     <- c("chatgpt", "gemini", "grammarly",
               "quillbot", "notion_ai", "phind",
               "edu_chat", "other")
use_cols  <- paste0("usefulness_", tools)

# Filter to adopters and compute “primary_tool”
df_adopt <- edx %>%
  # keep only rows where uses_ai_for_study == "True"
  filter(uses_ai_for_study == "True") %>%
  # replace NAs with a very low value so they aren’t picked
  mutate(across(all_of(use_cols), ~replace_na(.x, -Inf))) %>%
  # for each row, find which usefulness_* is maximal
  rowwise() %>%
  mutate(
    primary_tool = tools[which.max(c_across(all_of(use_cols)))]
  ) %>%
  ungroup() %>%
  # turn into a factor (caret needs this)
  mutate(primary_tool = factor(primary_tool, levels = tools))

# Train/test split on adopters (80/20 stratified by primary_tool)
set.seed(1)
idx        <- createDataPartition(df_adopt$primary_tool, p = 0.8, list = FALSE)
train_adopt <- df_adopt[idx, ]
test_adopt  <- df_adopt[-idx, ]


## ----CART tree----------------------------------------------------------------------------------------
# Fit a single CART tree with rpart
fit_tree <- rpart(
  uses_ai_for_study ~ age + gender + country + grade +
    uses_chatgpt + uses_gemini + uses_grammarly +
    uses_quillbot + uses_notion_ai + uses_phind +
    uses_edu_chat + uses_other,
  data    = train,
  method  = "class",
  control = rpart.control(cp = 0.01)
)


## ----Visualize the tree-------------------------------------------------------------------------------
# Visualize the tree
rpart.plot(
  fit_tree,
  type          = 0,    # node labels only (no split text under nodes)
  extra         = 0,    # no class/prob/count info in leaves
  fallen.leaves = TRUE, # align terminal nodes at the same depth
  cex           = 0.6   # smaller text so it never overlaps
)


## -----------------------------------------------------------------------------------------------------
ctrl      <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)
rpartGrid <- expand.grid(cp = seq(0.001, 0.1, length = 20))

set.seed(1)
train_rpart_cv <- train(
  uses_ai_for_study ~ age + gender + country + grade +
    uses_chatgpt + uses_gemini + uses_grammarly +
    uses_quillbot + uses_notion_ai + uses_phind +
    uses_edu_chat + uses_other,
  data      = train,
  method    = "rpart",
  metric    = "ROC",
  trControl = ctrl,
  tuneGrid  = rpartGrid
)

print(train_rpart_cv$bestTune)
plot(train_rpart_cv)


## -----------------------------------------------------------------------------------------------------
pred_tree <- predict(fit_tree, test, type = "class")
print(confusionMatrix(pred_tree, test$uses_ai_for_study))


## ----Random Forest------------------------------------------------------------------------------------
# Random Forest with caret tuning
rfGrid <- expand.grid(mtry = c(2, 4, 6, 8))

set.seed(1)
train_rf_cv <- train(
  uses_ai_for_study ~ age + gender + country + grade +
    uses_chatgpt + uses_gemini + uses_grammarly +
    uses_quillbot + uses_notion_ai + uses_phind +
    uses_edu_chat + uses_other,
  data       = train,
  method     = "rf",
  metric     = "ROC",
  ntree      = 500,
  trControl  = ctrl,
  tuneGrid   = rfGrid,
  importance = TRUE
)

print(train_rf_cv$bestTune)
plot(train_rf_cv)
pred_rf_cv <- predict(train_rf_cv, test)
print(confusionMatrix(pred_rf_cv, test$uses_ai_for_study))


## -----------------------------------------------------------------------------------------------------
#Cross-Validation and Tuning Setup

ctrl   <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = FALSE,    # multiclass accuracy doesn’t need probs
  summaryFunction = defaultSummary
)
rfGrid <- expand.grid(mtry = c(2, 4, 6, 8))

# Fit the multiclass Random Forest

set.seed(1)
rf_multi <- train(
  primary_tool ~ age + gender + country + grade +
    uses_chatgpt + uses_gemini + uses_grammarly +
    uses_quillbot + uses_notion_ai + uses_phind +
    uses_edu_chat + uses_other,
  data      = train_adopt,
  method    = "rf",
  metric    = "Accuracy",
  trControl = ctrl,
  tuneGrid  = rfGrid,
  ntree     = 500,
  importance= TRUE
)


## -----------------------------------------------------------------------------------------------------
# Inspect tuning results
print(rf_multi)            # shows accuracy by mtry
plot(rf_multi)             # visualizes the tuning curve


## -----------------------------------------------------------------------------------------------------
# Evaluate on the hold-out adopters
pred_multi <- predict(rf_multi, test_adopt)
cm_multi   <- confusionMatrix(pred_multi, test_adopt$primary_tool)
print(cm_multi)            # overall accuracy + per-class stats


## -----------------------------------------------------------------------------------------------------
# Variable importance
vi_multi <- varImp(rf_multi)
print(vi_multi)            # which features drive tool choice
plot(vi_multi, top = 10)   # plot the top 10 most important

