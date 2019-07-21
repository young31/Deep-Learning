# 데이터준비하기
library(keras)
library(readr)

data <- read_csv("download/spam.csv")

# 데이터 전처리
train_y = data$v1
train_x = data$v2
## 토크화과정에서 자꾸 에러가 났는데 포맷지정으로 해결
train_x = iconv(train_x, to = 'UTF-8')

# 데이터 전처리과정
max_features = 500
tokenizer = text_tokenizer(max_features)
tokenizer %>% fit_text_tokenizer(train_x)

seq_text = texts_to_sequences(tokenizer, train_x)

max_len = 100
train_x = seq_text %>% pad_sequences(maxlen = max_len)

# label은 spam과 ham으로 저장되어있는데 factor화해도 인식에러가 발생하여 
# 직접 수정함
for(i in 1:length(train_y)){
  if(train_y[i] == "spam"){
    train_y[i] = '0'
  } else train_y[i] = '1'
}

# RNN ONLY
# RNN을 stack할 경우 return_sequences를 신경써줘야함
model = keras_model_sequential()

model %>% layer_embedding(input_dim = 5572, output_dim = 32, input_length = 100) %>% 
  layer_gru(units = 32, activation = 'relu', recurrent_dropout = 0.2, return_sequences =  T) %>% 
  layer_gru(units = 32, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'softmax')


# 1D CNN
# RNN에 비해 비약적인 속도 향상을 경험할 수 있음
model = keras_model_sequential()
 
model %>% layer_embedding(input_dim = 5572, output_dim = 32, input_length = 100) %>% 
  layer_conv_1d(32, 5, activation = 'relu') %>% 
  layer_max_pooling_1d(3) %>% 
  layer_conv_1d(32, 5, activation = 'relu') %>%
  layer_max_pooling_1d(3) %>% 
  layer_conv_1d(32, 5, activation = 'relu') %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 2, activation = 'softmax')

# 1D CNN + RNN
# 적절한 조합으로 RNN이 너무 무겁다고 생각이 들 때 선택할 수 있는 방법
model = keras_model_sequential()

model %>% layer_embedding(input_dim = 5572, output_dim = 32) %>% 
  layer_conv_1d(32, 5) %>% 
  layer_max_pooling_1d(3) %>% 
  layer_gru(units = 32, activation = 'relu', recurrent_dropout = 0.2, return_sequences =  T) %>% 
  layer_gru(units = 32, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'softmax')

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'sparse_categorical_crossentropy',
  metrics = 'acc'
)

history = model %>% fit(
  train_x, train_y,
  epochs = 10,
  batchsize = 128,
  validation_split = 0.2
)

