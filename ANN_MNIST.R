
library(keras)

#연쇄방식, 함수형방식
model  = keras_model_sequential()

  #분류 ANN 파라미터
Nin = 784 #입력 계층의 노드 수 
Nh = 100 #은닉 계층의 노드 수
number_of_class = 10 #출력값이 가질 클래스 수
Nout = number_of_class #출력노드 수

  #모델 설정
  #ver.1
model %>%
  layer_dense(input_shape = c(Nin),
              activation = "relu",
              units = Nh) %>%
  layer_dense(units = Nh,
              activation = "relu") %>%
  layer_dense(units = Nout,
              activation = "softmax")

  #ver.2
model %>%
  layer_dense(units = Nh,
              activation = "relu",
              input_shape = c(Nin)) %>%
  layer_dense(units = Nout,
              activation = "softmax")

summary(model)

model %>%
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = "accuracy")

  #data input
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

  #0~9 target에 대해 one-hot encoder
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

  # reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
  # rescale
x_train <- x_train / 255
x_test <- x_test / 255

  #모델 학습
history <- model %>% fit(x_train, y_train, 
                         epochs = 5, batch_size = 100, 
                         validation_split = 0.2)

plot(history, y, metrics = NULL,
     method = c("auto", "ggplot2", "base"),
     smooth = getOption("keras.plot.history.smooth", TRUE),
     theme_bw = getOption("keras.plot.history.theme_bw", FALSE))

  #모델 평가
model %>% evaluate(x_test, y_test,
                   batch_size = 100)
