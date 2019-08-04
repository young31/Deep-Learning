library(keras)

data=dataset_mnist()

#c(c(train_x,train_y),c(test_X,test_y))
train_x=data$train$x
train_y=data$train$y
test_x=data$test$x
test_y=data$test$y

#c(num,y,x,channel) channel -> gray-1//color-3
train_x=array_reshape(train_x,c(60000,28,28,1))/255
test_x=array_reshape(test_x,c(10000,28,28,1))/255

#one-hot xx
#train_y=to_categorical(train_y)
#test_Y=to_categorical(test_y)

#for CV, set model 
build_model=function(){
  model = keras_model_sequential() %>% 
    layer_conv_2d(filters=32,kernel_size=c(3,3),activation='relu',input_shape=c(28,28,1)) %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>% 
    layer_conv_2d(filters=128,kernel_size=c(3,3),activation='relu') %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>% 
    layer_conv_2d(filters=128,kernel_size=c(3,3),activation='relu') %>% 
    layer_flatten() %>% 
    layer_dense(units=256,activation='relu') %>% 
    layer_dense(units=10,activation='softmax')
  
  model %>% compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=c('acc')
  )
}

#for CV, set data
k=10
indices=sample(1:nrow(train_x))
folds=cut(1:length(indices),breaks=k,labels = F)
loss = NULL
acc = NULL

for(i in 1:k){
  #partial for CV - consider data format(vec,matrix)
  val_indices=which(folds==i,arr.ind=T)
  
  val_x=array_reshape(train_x[val_indices,,,],c(length(val_indices),28,28,1))
  val_y=train_y[val_indices]
  
  partial_train_x=array_reshape(train_x[-val_indices,,,],c(c(nrow(train_x)-length(val_x)),28,28,1))
  partial_train_y=train_y[-val_indices]
  
  
  #get model preset
  model=build_model()
  
  #train model
  history=model %>% fit(partial_train_x,partial_train_y,epoch=10,batch_size=256,validation_data=list(val_x,val_y))
  
  #evaluate
  test=model %>% evaluate(test_x,test_y)
  loss[i]=test$loss
  acc[i]=test$acc
  results = cbind(loss, acc)
  # 한번에 작업시 오류나서 각자 구해서 합치기
}
#model %>% predict(test_x) -> each y's likelihood

results