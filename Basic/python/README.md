# how to use tensorboard
## 1. set callback

tb_hist = keras.callbacks.TensorBoard(log_dir='[<absolute dir path>]', histogram_freq=0, write_graph=True, write_images=True)

## 2. add-in fit function

fit(####,
    callbacks=[tb_hist]
    )
and fit model

## 3. call tensorboard

tensorboard --logdir=[<own graph dir>]  ## in my case, jupyter's home dir

if anyone knows better way, please tell me
i'm really really appreciate that
