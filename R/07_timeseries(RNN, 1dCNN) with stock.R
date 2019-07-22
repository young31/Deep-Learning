# 아쉬운 부분이 굉장히 많은 시계열 분석이다.
# 이동평균값을 참조하려다보니 데이터를 그만큼 잃는다.
# lag만큼의 값을 활용하려다보니 또 데이터를 잃는다.
# 이것 말고도 CV를 하자니 참조할 데이터들이 겹친다. CV의 핵심은 완전분리라고 생각하거늘..
# 진행하는 중 가장 부족한 부분이 아닐까 싶다. 

library(keras)
library(TTR)
library(readr)

data <- read_csv("download/004170.csv")
data=as.data.frame(data)

# moving average // TTR package
for(i in c(5,10,30,60)){
  p <- (SMA(x = data$open, n = i))
  data = cbind(data,p)
  }

for(i in c(5,10,30,60)){
  vol <- (SMA(x = data$vol, n = i))
  data = cbind(data,vol)
}

for(i in c(5,10,30,60)){
  per <- (SMA(x = data$per, n = i))
  data = cbind(data,per)
}

for(i in c(5,10,30,60)){
  pbr <- (SMA(x = data$pbr, n = i))
  data = cbind(data,pbr)
}

tar_data = data[,9:24]
names(tar_data)=c('moving_p5','moving_p10','moving_p30','moving_p60',
                  'moving_vol5','moving_vol10','moving_vol30','moving_vol60',
                  'moving_per5','moving_per10','moving_per30','moving_per60',
                  'moving_pbr5','moving_pbr10','moving_pbr30','moving_pbr60')

#except NA data
int_data = cbind(tar_data,data$open)
int_data = int_data[61:nrow(int_data),]

# divide data
sn = round(nrow(int_data)*0.7)
train_data = int_data[1:sn,]
val_data = int_data[(sn+1):nrow(int_data),]

#preprocessing
mean = apply(train_data,2,mean,na.rm=T)
std = apply(train_data,2,sd,na.rm=T)
train_data = scale(train_data, center = mean, scale = std)

mean = apply(val_data,2,mean,na.rm=T)
std = apply(val_data,2,sd,na.rm=T)
val_data = scale(val_data, center = mean, scale = std)


train_x = train_data[,1:16]
train_y = train_data[,17]

val_x = val_data[,1:16]
val_y = val_data[,17]


#lags
n = 30

#make array shape(3d tensor shape)
train_x = array(data = lag(train_x, n)[-(1:n), ],  dim = c((nrow(train_x) - n), n, ncol(train_x)))
train_y = train_y[(n+1):length(train_y)]
#y_re = array(data = y[-(1:lags)], dim = c(nrow(y)-lags, 1))

val_x = array(data = lag(val_x, n)[-(1:n), ],  dim = c((nrow(val_x) - n), n, ncol(val_x)))
val_y = val_y[(n+1):length(val_y)]

# RNN
model = keras_model_sequential()
model %>% layer_gru(units=128,return_sequences = T,input_shape = c(n,16)) %>% 
  layer_gru(units=128, return_sequences = F) %>% 
  layer_dense(units=1,activation = 'linear')

# bidirectional-- 
model = keras_model_sequential()
model %>% layer_gru(units = 128, input_shape = c(n, 16),return_sequences = T) %>% 
  bidirectional(layer_gru(units=128), input_shape = c(n,16)) %>%
  layer_dense(units=1,activation = 'linear')

# 1D CNN
model = keras_model_sequential()
model %>% layer_conv_1d(filters = 32, input_shape = list(n, 16), kernel_size = 3, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 2) %>% 
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 2) %>% 
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu') %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(1)




model %>% compile(optimizer = 'rmsprop',
                  loss = 'mse',
                  metrics = 'mae')

histroy = model %>% fit(train_x, train_y,
                        batch_size = 128,
                        epochs = 10,
                        validation_data=list(val_x, val_y)
                        )

# 모형의 성능은 bidirection이 다소 우수하나 최적화가 완전히 되지 않았다는 점을 기억하자.
# 시계열 자료에서 CNN의 성능은 다소 떨어진다. 참조할 변수가 그리 크지 않은 것이 문제일 수도 있다.
# 시간은 text자료에 비해 월등히 짧다. 비정형데이터들의 어려움을 알 수 있었다.

