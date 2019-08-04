가장 활용이 많이 되는 CNN중 한 가지입니다.

여러가지 버전이 있지만 기본적인 외형은 여러가지 방향으로 풀었다 다시 묶는 그림으로 기억에 남습니다.

그림으로 보면 이해하기 쉽습니다.

![incepotion01](C:\Users\young\Desktop\git\Deep-Learning\Architecture\images\incepotion01.png)



위 와 같은 stem을 여러개 연결시켜 모형을 구성해 갑니다.

![inception02](C:\Users\young\Desktop\git\Deep-Learning\Architecture\images\inception02.png)



버전이 올라가면서 주축이 되는 점은 연산량과 속도에 대한 문제입니다.  

5x5 이상의 Conv창을 여러개의 3x3창으로 교체하면서 연산을 줄이기도 하고, 1xN, Nx1의 형태로 변환시키는 방법을 사용하기도 합니다.  

이를 factorizing이라고 합니다.  

inception모듈은 이후 다양하게 활용되었는데 이는 추후 알아보도록 하겠습니다.

마지막으로 한 층을 코드로 구현해보면 다음과 같습니다.

```python
from keras import models, layers

input_layer = layers.Input(shape=(size,size,channel))

branch_a = layers.Conv2D(filters=24, kernel_size=1, activation='relu')(input_layer)
branch_a = layers.AveragePooling2D(1, strides=2)(branch_a)

branch_b = layers.Conv2D(filters=16, kernel_size=1, activation='relu',strides=2)(input_layer)

branch_c = layers.Conv2D(filters=16, kernel_size=1, activation='relu', strides=2)(input_layer)
branch_c = layers.Conv2D(filters=24, kernel_size=3, activation='relu', padding='same')(branch_c)


branch_d = layers.Conv2D(filters=16, kernel_size=1, activation='relu', strides=2)(input_layer)
branch_d = layers.Conv2D(filters=24, kernel_size=3, activation='relu', padding='same')(branch_d)
branch_d = layers.Conv2D(filters=24, kernel_size=3, activation='relu', padding='same')(branch_d)

output_1 = layers.concatenate([branch_a, branch_b, branch_c, branch_d])

out_branch_a = layers.Conv2D(filters=16, kernel_size=1, activation='relu', strides=2)(output_1)
out_branch_a = layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(out_branch_a)

out_branch_b = layers.Conv2D(filters=16, kernel_size=1, activation='relu', strides=2)(output_1)

output_2 = layers.concatenate([out_branch_a, out_branch_b])

batch_normal = layers.BatchNormalization()(output_2)
global_average_pooling = layers.GlobalAveragePooling2D()(batch_normal)
output = layers.Dense(units=classes, activation='softmax')(global_average_pooling)

model = keras.Model(input_layer, output)
```

