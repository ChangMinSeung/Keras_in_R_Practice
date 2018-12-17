
library(keras)
library(tibble)

#연쇄방식, 함수형방식

  #회귀 ANN 파라미터
Nin = 13 #입력 벡터의 길이(컬럼의 수)
Nh = 5 #은닉 계층의 수
Nout = 1 #출력 계층의 길이

  #모델 설정
build_model <- function() { 
  
  model = keras_model_sequential() %>%
          layer_dense(units = Nh,
                      input_shape = c(Nin),
                      activation = "relu") %>%
          layer_dense(units = Nh,
                      activation = "relu") %>%
          layer_dense(units = Nout)

  model %>% compile(
                    loss = "mse",
                    optimizer = "sgd",
                    metrics = "accuracy"
                    )

  model
}

model <- build_model()
model %>% summary()

  #data input
boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
train_df <- as_tibble(train_data)
colnames(train_df) <- column_names

head(train_df)

  #Normalize features
  #Test data is *not* used when calculating the mean and std.
train_data <- scale(train_data) 

col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")

test_data <- scale(test_data, 
                   center = col_means_train, 
                   scale = col_stddevs_train)

  #모델 학습 #verbose:자세한내용모드
history <- model %>% fit(train_data, train_labels, 
                         epochs = 100, batch_size = 100, 
                         validation_split = 0.2, verbose = 2) 

plot(history, y, metrics = NULL,
     method = c("auto", "ggplot2", "base"),
     smooth = getOption("keras.plot.history.smooth", TRUE),
     theme_bw = getOption("keras.plot.history.theme_bw", FALSE))

  #모델 평가
model %>% evaluate(test_data, test_labels,
                   batch_size = 100)
