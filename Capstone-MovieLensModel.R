##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)
​
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
​
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
​
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
​
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
​
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
​
movielens <- left_join(ratings, movies, by = "movieId")
​
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
​
# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
​
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
​
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##############################################################
#Capstone Movielens Model
##############################################################

#Partitioning Test and Train
set.seed(1901, sample.kind = "Rounding")

indexes <- split(1:nrow(edx), edx$userId)
test_ind_capstone <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) %>%
  unlist(use.names = TRUE) %>% 
  sort()
test_set_capstone <- edx[test_ind_capstone,]
train_set_capstone <- edx[-test_ind_capstone,]

#Removing movies that are not in both test & train set
test_set_capstone <- test_set_capstone %>% 
  semi_join(train_set_capstone, by = "movieId")
train_set_capstone <- train_set_capstone %>% 
  semi_join(test_set_capstone, by = "movieId")

#pivot wider to get matrix
y <- select(train_set_capstone, movieId, userId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

#RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Calculating Rating Mean
mu<-mean(y, na.rm = TRUE)
mu
#Modeling Movie Effects
b_i <- colMeans(y - mu, na.rm = TRUE)
fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

left_join(test_set_capstone, fit_movies, by = "movieId") %>% 
  mutate(pred = mu + b_i) %>% 
  summarize(rmse = RMSE(rating, pred))

#Modeling User Effects
b_u <- rowMeans(y, na.rm = TRUE)
b_u <- rowMeans(sweep(y - mu, 2, b_i), na.rm = TRUE)

fit_users <- data.frame(userId = as.integer(rownames(y)), b_u = b_u)

left_join(test_set_capstone, fit_movies, by = "movieId")%>% 
  left_join(fit_users, by = "userId")%>%
  mutate(pred = mu + b_i + b_u)%>%
  summarize(rmse = RMSE(rating, pred))

#Modeling Genre Effects
b_g <- train_set_capstone %>%
  left_join(fit_movies, by = "movieId") %>%
  left_join(fit_users, by = "userId") %>%
  mutate(resid = rating - mu - b_i - b_u) %>%
  separate_rows(genres, sep= "\\|")%>%
  group_by(genres) %>%
  summarize(b_g = mean(resid, na.rm = TRUE))

left_join(test_set_capstone, fit_movies, by = "movieId") %>%
  left_join(fit_users, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0),
    pred = mu + b_i + b_u + b_g) %>%
  summarize(rmse = RMSE(rating, pred))

#Modeling Time stamp Effects
library(lubridate)

train_set_capstone <- train_set_capstone %>%
  mutate(time_unit = year(as_datetime(timestamp)))

b_t <- train_set_capstone %>%
  left_join(fit_movies, by = "movieId") %>%
  left_join(fit_users, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(resid = rating - mu - b_i - b_u - b_g) %>%
  group_by(time_unit) %>%
  summarize(b_t = mean(resid, na.rm = TRUE))

test_set_capstone <- test_set_capstone %>%
  mutate(time_unit = year(as_datetime(timestamp)))

left_join(test_set_capstone, fit_movies, by = "movieId") %>%
  left_join(fit_users, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "time_unit") %>%
  mutate(
    b_i = replace_na(b_i, 0),
    b_u = replace_na(b_u, 0),
    b_g = replace_na(b_g, 0),
    b_t = replace_na(b_t, 0),
    pred = mu + b_i + b_u + b_g + b_t
  ) %>%
  summarize(rmse = RMSE(rating, pred))

#Finding Best Lambda for Regularizing Movie Effects & User Effects & Genre Effects
lambdas <- seq(0, 10, 0.5)
rmses<-sapply(lambdas, function(l) {
  b_i_reg <- train_set_capstone %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
 
  b_u_reg <- train_set_capstone %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + l))
  
  b_g_reg <- train_set_capstone %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + l))
  
  b_t_reg <- train_set_capstone %>%
    mutate(time_unit = year(as_datetime(timestamp))) %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by = "genres") %>%
    group_by(time_unit) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u - b_g) / (n() + l))

  test_set_capstone %>%
    mutate(time_unit = year(as_datetime(timestamp))) %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by = "genres") %>%
    left_join(b_t_reg, by = "time_unit") %>%
    mutate(b_g = replace_na(b_g, 0),
           b_t = replace_na(b_t, 0),
           pred = mu + b_i + b_u + b_g + b_t) %>%
    summarize(rmse = RMSE(rating, pred)) %>%
    pull(rmse)
})

best_lambda <- lambdas[which.min(rmses)]
best_lambda
min(rmses)

#Regularization using best Lambda
b_i_reg <- train_set_capstone %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + best_lambda))

b_u_reg <- train_set_capstone %>%
  left_join(b_i_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + best_lambda))

b_g_reg <- train_set_capstone %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + best_lambda))

b_t_reg <- train_set_capstone %>%
  mutate(time_unit = year(as_datetime(timestamp))) %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  group_by(time_unit) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u - b_g) / (n() + best_lambda))

#Testing the Regularization (& Replacing NAs with zeros)
rmse_reg_all <- test_set_capstone %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  left_join(b_t_reg, by = "time_unit") %>%
  mutate(
    b_i = replace_na(b_i, 0),
    b_u = replace_na(b_u, 0),
    b_g = replace_na(b_g, 0),
    b_t = replace_na(b_t, 0),
    pred = mu + b_i + b_u + b_g + b_t
  ) %>%
  summarize(rmse = RMSE(rating, pred)) %>%
  pull(rmse)

rmse_reg_all

#Testing it with the final holdout 
final_holdout_test<- final_holdout_test %>%
  mutate(time_unit = year(as_datetime(timestamp)))

final_rmse <- final_holdout_test %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  left_join(b_t_reg, by = "time_unit") %>%
  mutate(
    b_i = replace_na(b_i, 0),
    b_u = replace_na(b_u, 0),
    b_g = replace_na(b_g, 0),
    b_t = replace_na(b_t, 0),
    pred = mu + b_i + b_u + b_g + b_t ) %>%
  summarize(rmse = RMSE(rating, pred)) %>%
  pull(rmse)

final_rmse
