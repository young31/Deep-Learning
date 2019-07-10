library(keras)

rt=dataset_reuters(num_words = 10000)

c(c(train_x,train_y),c(test_x,test_y))%<-%rt

vec_seq=function(seq,dim=10000){
  results=matrix(0,nrow=length(seq),ncol=10000)
  for(i in 1:length(seq))
    results[i,seq[[i]]]<-1
  results
}

train_x=vec_seq(train_x)
test_x=vec_seq(test_x)

one_hot=function(lab,dimension=46){
  results=matrix(0,nrow=length(lab),ncol=46)
  for(i in 1:length(lab))
    results=results[i,lab[[i]]+1]<-1
  results
}

train_y=to_categorical(train_y)
test_y=to_categorical(test_y)

model_build=function(){
  
  model=keras_model_sequential()
  
  model %>% 
    layer_dense(units=64,activation='relu',input_shape=c(10000)) %>%
    layer_dense(units=64,activation='relu') %>% 
    layer_dense(units=46,activation='softmax')
  
  model %>% compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=c('acc')
  )
}

model = model_build()

hisory=model %>% fit(train_x,train_y,batch_size=128,epoch=10,validation_split=0.2)

model %>% evaluate(test_x,test_y)


## CV
k=4
indices=sample(1:nrow(train_x))
folds=cut(1:length(indices),breaks=k,labels = F)
results=NULL

for(i in 1:k){
  val_indices=which(folds==i,arr.ind=T)
  
  partial_train_x=train_x[-val_indices,]
  partial_train_y=train_y[-val_indices,]
  
  val_x=train_x[val_indices,]
  val_y=train_y[val_indices,]
  
  model=model_build()
  
  history=model %>% fit(partial_train_x,partial_train_y,epoch=10,batch_size=512,validation_data=list(val_x,val_y))
  
  test=model %>% evaluate(test_x,test_y)
  results[i]=test$acc
}

results
