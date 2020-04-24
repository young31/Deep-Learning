library(keras)

data_from_dir='C:/Users/young/Desktop/R/cats-dogs'
base_dir='C:/Users/young/Desktop/R/kaggle_cat'

train_dir=file.path(base_dir,'train')
val_dir=file.path(base_dir,'val')
test_dir=file.path(base_dir,'test')

train_data_gen=image_data_generator(rescale = 1/255, zoom_range = 0.2)
val_data_gen=image_data_generator(rescale = 1/255)

train_gen=flow_images_from_directory(
  train_dir,
  train_data_gen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = 'binary'
)

val_gen=flow_images_from_directory(
  val_dir,
  val_data_gen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = 'binary'
)

test_gen=flow_images_from_directory(
  test_dir,
  val_data_gen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = 'binary'
  
)
conv_vgg16 = application_vgg16(
  weights = 'imagenet',
  include_top = F,
  input_shape = c(150,150,3)
)

conv_vgg16

model = keras_model_sequential()
model %>% conv_vgg16 %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model

model %>% compile(optimizer=optimizer_rmsprop(lr=0.00001),
                  loss='binary_crossentropy',
                  metrics='acc')


freeze_weights(conv_vgg16)

history = model %>% fit_generator(
  train_gen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = val_gen,
  validation_steps = 50
)


## fine-tuning
conv_vgg16

unfreeze_weights(conv_vgg16, from='block3_convl')

model %>% compile(
  optimizer=optimizer_rmsprop(lr=1e-5),
  loss='binary_crossentropy',
  metrics='acc'
)

history = model %>% fit_generator(
  train_gen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = val_gen,
  validation_steps = 50
)
model %>% evaluate_generator(test_gen,steps = 50)
