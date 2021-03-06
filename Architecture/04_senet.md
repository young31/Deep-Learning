res-net이 나온 이후에 오차를 더 낮추기 위해 나온 방안 중 하나입니다. 

SE-net의 SE는 각각 Squeeze와 Excitation을 의미합니다. 이는 특징들을 잘 추출하여 재조정을 통해 더 잘 반영하겠다는 생각입니다.

squeeze 단계에서는 추출을 위해 global average pooling을 사용합니다. 이를 통해 크기를 (1, 1, c)로 만드는 작업을 합니다. 이 후 excitation단계에서는 FC층을 통하여 재조정합니다. (이 부분을 보면서 AE가 생각났습니다. ))

![senet01](images/senet01.png)

이렇게 조정된 특징들을 다음 층으로 넘겨줍니다. 이 과정에서 res-net과 함께 사용될 수 있는 것을 보아 여러모로 변형하여 적용시켜 볼 수 있을 것 같습니다. 

![senet02](images/senet02.png)

res-net과 se-net을 알아보면서 공통적인 점은 신경망의 구조를 향상시킬 때 기본적인 방향외의 우회로를 통해 정보를 반영시킨다는 것입니다. 신경망이 깊어지면서 일어나는 손실을 보완하기 위해 어떤 정보를 어떠한 방법으로 반영할지를 생각해보아야겠습니다. 
