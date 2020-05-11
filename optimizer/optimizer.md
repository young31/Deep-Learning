머신러닝의 여러 기법들은 손실함수를 정의하고 이를 지속적으로 감소시키는 방향으로 학습을 진행한다.이 때 손실함수의 최수값을 찾아가는 방법에 대한 것이 optimizer다.

이에 대해서는 많은 연구들이 이루어졌다. 목표를 정확히 찾아가면서 속도를 빠르게 하는 방법이 연구들의 핵심이었고 아래는 관련하여 간략하게 정리한 사진이다.

![opt01](images/opt01.jpg)

사진에서 알 수 있듯이 최소값을 찾아가기 위해서 어떤 방향을 탐색해서 갈 것인지, 얼만큼 내려가야할 것인지에 대한 부분들에 대해서 다양한 방식으로 연구가 진행되어왔다. 

그 중 사진에서 볼 수 있는 adam아고리즘은 잘 모를 때는 adam을 사용하라는 말이 있을 정도로 대중적으로 많이 사용되는 알고리즘인것 같다.

그 외에도 다양한 알고리즘들은 keras에 이미 구현되어 쉽게 사용할 수 있다.  케라스의 공식문서를 통해 자주 사용되는 optimizer에 대해 활용법을 알아보자

- adadelta: 파라미터로는 lr, rho, epsilon이 있으며 그대로 사용하는 것이 추전된다고 한다. 

  논문을 참조해서 보면 학습률은 자동으로 업데이트 되므로 rho만 조절하여 사용하면 될 것 같다.

- SGD: 파라미터로는 lr, momentum, decay, nesterov가 있으며 momentum 항을 조절하여 사용하면 될 것이다. 

  nesterov는 불옵션으로 약간 변형한 알고리즘을 적용할 수 있다.

- rmsprop: 파라미터로는 lr, rho, epsilon이 있으며 변경하지 않고 사용하는 것이 추천된다고 한다. 

- adam: 파라미터로 lr, beta_1, beta_2, epsilon이 있으며 default값은 논문에서 제안한 beta_1 = 0.9, beta_2 = 0.999로 설정되어 있다. 

  ​	

  자세한 사항은 [keras공식 문서](http://faroit.com/keras-docs/0.2.0/optimizers/)를 참조하거나, [여기](https://forensics.tistory.com/28)에서 수식과 관련 코드를 볼 수 있다.

  

  

마지막으로 각 옵티마이저가 길을 찾아가는 것을 시각적으로 표현한 것이 있어 확인해 보면 도움이 될 것 같다.

![opt2](images/opt2.gif)

![opt3](images/opt3.gif)