library(keras)

data_from_dir='C:/Users/young/Desktop/R/cats-dogs'
base_dir='C:/Users/young/Desktop/R/kaggle_cat'
dir.create(base_dir)


#dir.create
train_dir=file.path(base_dir,'train')
val_dir=file.path(base_dir,'val')
test_dir=file.path(base_dir,'test')

train_cats_dir=file.path(train_dir,'cats')
train_dogs_dir=file.path(train_dir,'dogs')
val_cats_dir=file.path(val_dir,'cats')
val_dogs_dir=file.path(val_dir,'dogs')
test_cats_dir=file.path(test_dir,'cats')
test_dogs_dir=file.path(test_dir,'dogs')

# copy file
fnames=paste0('cat.',1:1000,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(train_cats_dir))
fnames=paste0('dog.',1:1000,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(train_dogs_dir))
fnames=paste0('cat.',1001:1500,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(val_cats_dir))
fnames=paste0('dog.',1001:1500,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(val_dogs_dir))
fnames=paste0('cat.',1501:2000,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(test_cats_dir))
fnames=paste0('dog.',1501:2000,'.jpg')
file.copy(file.path(data_from_dir,fnames),
          file.path(test_dogs_dir))

# make generator
data_gen=image_data_generator(rescale = 1/255)

train_gen=flow_images_from_directory(
  train_dir,
  data_gen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = 'binary'
)

val_gen=flow_images_from_directory(
  val_dir,
  data_gen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = 'binary'
)

# build model
model=keras_model_sequential()

model %>% 
  layer_conv_2d(filters = 32,activation='relu',input_shape = c(150,150,3),kernel_size = c(3,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filter=64,activation='relu',kernel_size = c(3,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filter=64,activation='relu',kernel_size = c(3,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filter=128,activation='relu',kernel_size = c(3,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_flatten() %>% 
  layer_dense(units=512,activation='relu') %>% 
  layer_dense(units = 1,activation='sigmoid')

model

model %>% compile(
  optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=c('acc')
)

# run/save/load model
history=model %>% fit_generator(
  train_gen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = val_gen,
  validation_steps = 50
)

model %>% save_model_hdf5('cats_dogs.h5')
# m = load_model_hdf5('C:/Users/young/Desktop/R/cats_dogs.h5')



# visualizing how to work each nodes
## take one image
img = image_load('kaggle_cat/test/cats/cat.1601.jpg', target_size = c(150, 150)) %>% 
  image_to_array() %>% 
  array_reshape(c(1,150,150,3))/255

plot(as.raster(img[1,,,]))

## take each nodes
layer_outputs = lapply((model$layers[1:8]), function(layer) layer$output)
activation_model = keras_model(inputs = model$input, outputs = layer_outputs)

activations = activation_model %>% predict(img)

## define function to plot result
plot_channel = function(channel){
  rotate = function(x) t(apply(x,2,rev))
  image(rotate(channel),axes=F, asp=1, col=terrain.colors(16))
}

str(activations)

## compare each nodes' result
par(mfrow=c(2,3))
plot_channel(activations[[1]][1,,,1])
plot_channel(activations[[1]][1,,,7])
plot_channel(activations[[1]][1,,,10])
plot_channel(activations[[3]][1,,,1])
plot_channel(activations[[3]][1,,,7])
plot_channel(activations[[3]][1,,,10])

