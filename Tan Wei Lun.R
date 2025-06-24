## ----setup, include=FALSE-------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(caret)
library(tidyr)
library(dplyr)
library(ggplot2)


## ----datasets-------------------------------------------------------------------------------------------------------------------------------------------
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



## ----Dimensions of the edx Dataset----------------------------------------------------------------------------------------------------------------------
dim(edx)


## ----Frequency of All Ratings---------------------------------------------------------------------------------------------------------------------------
edx %>%
  filter(rating >= 0 & rating <= 5) %>%
  count(rating) %>%
  arrange(rating)


## -----The histogram is create to visualize the rating distribution--------------------------------------------------------------------------------------
edx %>%
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  scale_x_continuous(breaks = seq(0, 5, 0.5)) +
  labs(title = "Distribution of Movie Ratings",
       x = "Rating",
       y = "Count")


## ---Number of Unique Movies and Users-------------------------------------------------------------------------------------------------------------------
n_distinct(edx$movieId)
n_distinct(edx$userId)


## ---Histograms of the number of ratings per movie and per user------------------------------------------------------------------------------------------
# Ratings Per Movie (Log Scale)
edx %>%
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  scale_x_log10() +
  labs(title = "Distribution of Number of Ratings per Movie (Log Scale)",
       x = "Number of Ratings",
       y = "Number of Movies")

# Ratings Per User (Log Scale)
edx %>%
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  scale_x_log10() +
  labs(title = "Distribution of Number of Ratings per User (Log Scale)",
       x = "Number of Ratings",
       y = "Number of Users")


## -----Number of Ratings for All Genres------------------------------------------------------------------------------------------------------------------
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres) %>%
  arrange(desc(n))


## ------Bar plot for top 10 most frequent genres---------------------------------------------------------------------------------------------------------
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres) %>%
  top_n(10, n) %>%
  ggplot(aes(x = reorder(genres, n), y = n)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 10 Genres by Number of Ratings",
       x = "Genre",
       y = "Number of Ratings")


## ---Half-Star vs Whole-Star Ratings---------------------------------------------------------------------------------------------------------------------
edx %>%
  mutate(rating_type = ifelse(rating %% 1 == 0, "Whole-Star", "Half-Star")) %>%
  count(rating_type)


## ----bar plot for “Whole-Star” and “Half-Star” categories-----------------------------------------------------------------------------------------------
edx %>%
  mutate(rating_type = ifelse(rating %% 1 == 0, "Whole-Star", "Half-Star")) %>%
  count(rating_type) %>%
  ggplot(aes(x = rating_type, y = n, fill = rating_type)) +
  geom_col() +
  labs(title = "Comparison: Whole-Star vs Half-Star Ratings",
       x = "Rating Type", y = "Count") +
  theme(legend.position = "none")


## ---RMSE Function---------------------------------------------------------------------------------------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## ---Dataset Split for Model Testing--------------------------------------------------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index, ]
edx_test <- edx[test_index, ]

# Keep only users and movies in test that are also in train
edx_test <- edx_test %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")


## ---Baseline Model-------------------------------------------------------------------------------------------------------------------------------------
mu_hat <- mean(edx_train$rating)

naive_rmse <- RMSE(edx_test$rating, mu_hat)
naive_rmse


## ---Movie Effect Model----------------------------------------------------------------------------------------------------------------------------------
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu_hat + b_i) %>%
  pull(pred)


## ----RMSE for Movie Effect Model------------------------------------------------------------------------------------------------------------------------
movie_effect_rmse <- RMSE(edx_test$rating, predicted_ratings)
movie_effect_rmse


## ------Movie + User Effect Model------------------------------------------------------------------------------------------------------------------------
user_avgs <- edx_train %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)


## -------RMSE for Movie + User Effect Model--------------------------------------------------------------------------------------------------------------
movie_user_effect_rmse <- RMSE(edx_test$rating, predicted_ratings)
movie_user_effect_rmse


## ------Regularized Movie + User Effect Model------------------------------------------------------------------------------------------------------------
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(lambda) {
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  predicted_ratings <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})

# Best lambda
best_lambda <- lambdas[which.min(rmses)]
best_lambda


## -----Graph for RMSE versus lambda----------------------------------------------------------------------------------------------------------------------
ggplot(data = data.frame(lambda = lambdas, rmse = rmses), aes(x = lambda, y = rmse)) +
  geom_point() +
  labs(title = "RMSE vs Lambda", x = "Lambda", y = "RMSE")


## ------RMSE for Regularized Movie + User Effect Model---------------------------------------------------------------------------------------------------
mu <- mean(edx_train$rating)

b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + best_lambda))

b_u <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + best_lambda))

predicted_ratings <- edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

regularized_rmse <- RMSE(edx_test$rating, predicted_ratings)
regularized_rmse


## ---Summary Table---------------------------------------------------------------------------------------------------------------------------------------
rmse_results <- tibble(
  method = c("Naive Mean", "Movie Effect", "Movie + User Effect", "Regularized Movie + User"),
  RMSE = c(naive_rmse, movie_effect_rmse, movie_user_effect_rmse, regularized_rmse)
)

print(rmse_results)


## ----Final Evaluation on Holdout Set--------------------------------------------------------------------------------------------------------------------
lambda <- best_lambda  

# Global average
mu <- mean(edx$rating)

# Movie effect
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

# User effect
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

# Predict final ratings
final_predictions <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Replace any missing predictions with mu 
final_predictions[is.na(final_predictions)] <- mu

# Final RMSE
final_rmse <- RMSE(final_holdout_test$rating, final_predictions)
final_rmse

