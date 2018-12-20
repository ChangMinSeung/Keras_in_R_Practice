
library(keras)

  #파라미터
epochs = 10
batch_size = 128

  #input image dimensions
img_rows <- 28
img_cols <- 28

  #data input
mnist <- dataset_mnist()
c(x_train, y_train) %<-% mnist$train
c(x_test, y_test) %<-% mnist$test

  #Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

  #Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

  #Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

  #Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, #합성곱
                kernel_size = c(3,3), 
                activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% #2차원 이미지를 1차원 벡터로 변환
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

summary(model)
  
model %>% compile(loss = loss_categorical_crossentropy,
                  optimizer = optimizer_adadelta(),
                  metrics = c('accuracy'))

  #Train model
model %>% fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              validation_split = 0.2)

scores <- model %>% evaluate(x_test, y_test, 
                             verbose = 0)

  #Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')