library(keras)
data=dataset_mnist()

train_x=data$train$x
train_y=data$train$y
test_x=data$test$x
test_y=data$test$y

train_x=array_reshape(train_x,c(60000,28*28))/255
test_x=array_reshape(test_x,c(10000,28*28))/255


model=keras_model_sequential() %>% 
  layer_dense(units=256,activation='relu',input_shape=c(28*28)) %>%
  layer_dropout(rate=0.2) %>% 
  layer_dense(units=128,activation='relu') %>% 
  layer_dropout(rate=0.2) %>% 
  layer_dense(128,activation='relu') %>% 
  layer_dense(units=10,activation='softmax')

model %>% compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=c('accuracy')
)

history=model %>% fit(train_x,train_y,epoch=10,batch_size=512,validation_data=list(test_x,test_y))