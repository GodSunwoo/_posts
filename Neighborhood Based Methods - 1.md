# Neighborhood Based Methods - 1



​	Neighborhood Based 방법은 가장 간단한 방법입니다. 사람이든, 아이템이든, 가장 가까운 사람이나 아이템을 매칭해서 추천하는 방식이죠. Neighborhood 방식은 크게 User Based 방법과 Item Based 방법입니다. 구체적인 아이디어와 구현방식은 이어서 다루도록 하겠습니다.

​	본 챕터에서는 유저의 관측되지 않은 데이터의 빈칸을 채우는것을 그 목표로 합니다. 일종의 Imputation이나 Prediction의 개념으로 이해하면 좋을 것 같습니다. 

### 1. Similarity Measure

​	User Based Model의 가장 큰 특징은, 유저의 유사성을 토대로 빈칸을 추론한다는 점입니다. a라는 유저와 가장 유사한 유저 $a^*$를 뽑아서 그 유저들의 Rating으로 a라는 유저의 빈 값들을 Impute하는 방식을 채택합니다. 사실 이는 굉장히 직관적이고 간단한 방법이지만, 모델을 해석할 수 있다는 부분에서 강점을 갖습니다. 

​	다음과 같은 User-Item Matrix를 생각해보겠습니다.

![image-20201231132651204](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231132651204.png)

​	다음 빈칸을 예측하는 것은 Cosine Similarity나 Pearson Correlation을 주로 사용합니다. 

- Cosine Similarity

![image-20201231133200237](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231133200237.png)

- Pearson Correlation

![Python - Pearson Correlation Test Between Two Variables - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20200311233526/formula6.png)

​	위 둘 모두 이들의 관계성 측도를 파악하는데 활용할 수 있습니다. 제가 생각한 이 측도의 특징은, 두 유저의 선형적 관계성을 탐색하는데 그친다는 것입니다. 2차원 이상의 비선형적 패턴은 이 측도로 탐색할 수 없습니다. 

​	이 두 측도를 사용하여, 두 유저의 판단합니다. 

![image-20201231134026689](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231134026689.png)



​	간략하게 표현하면 위와 같은 수식이 유도되겠네요. 가령 유저 1에 대해서, 위 수식을 접목시키면 $\mu_1 = \frac{7+6+7+4+5+4}{6} = 5.5, ~~\mu_3 = \frac{3+3+1+1}{4} = 2$ 과 같이 어렵지 않게 구할 수 있습니다. 1번 유저와 3번 유저를 잘 살펴보시면, 아이템 2 3 4 5를 둘 다 구매한것을 확인할 수 있습니다. 이에 따라서 이 4개의 아이템을 공통값으로 하여 둘의 유사성을 추정할 수 있겠네요. 

​	이 $\mu_u$를 계산하는 방식도 다양하게 존재합니다. 해당 유저의 관측된 모든 값을 다 넣을수도 있고, Target User와 교집합에 해당하는 아이템만 넣을수도 있습니다. 하지만 Sparse한 데이터 특성을 고려하면, 대부분 전자의 경우를 활용합니다.

​	여기서, 코사인 유사도와 피어슨 상관계수의 차이를 한번 짚고 넘어가겠습니다. 코사인 유사도는 표본평균값을 빼주지 않고, 피어슨씨는 그것을 빼줍니다. 일종의 Centered 개념입니다. 어떤 유저의 경우 평이 좀 후해서 전체적으로 높은 값을 주는 유저가 있고 (유저 1) 어떤 유저는 평이 좀 박해서 짠 값을 주는 유저가 있죠. (유저 4) 이러한 피어슨 상관계수는 개인별 편차를 보정해 주지만 코사인 유사도는 그러한 Scaling은 해주지 못합니다. 물론 피어슨상관계수도 완벽한 측도는 아닙니다. 

​	이를통해 구체적으로 계산을 해보면 다음과 같습니다.

![image-20201231134926502](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231134926502.png)

​	물론 이 유사도는 매우 큰 값일수도, 작은 값일수도 있습니다. 상관계수 값이 0.9라고 무조건 이는 큰 값이 아닙니다. 사실 Utility Matrix는 굉장히 Sparse해서 (대부분 0으로 차있어서) 두 유저가 겹쳐지는 값이 굉장히 적을것입니다. 따라서 이러한 점을 감안하여 항상 상대적으로 평가해야합니다. 



### 2. Prediction

​	이제 이 값을 활용해서 어떻게 빈 값을 채우는지 확인해 보겠습니다. Imputation을 위해서 우리는 가중평균을 활용합니다. . 하지만 우리는 모든 유저에 대한 데이터를 가중평균할 필요는 결코 없습니다. 왜냐하면 그렇게 진행할 경우 유사하지 않은 유저들의 데이터까지 입력으로 들어오게 되기 때문입니다. 따라서 우리는 상위 k개명의 유저의 데이터만 가중평균하면 됩니다. 그 k의 값은 hyper parameter로 우리가 임의로 설정할 수 있습니다. 이들의 집합을 $P_u (j)$ . 즉 유저 u에게 item j를 추천하기 위해 출력된 k명의 유저집합입니다. 

​	우리는 두가지 경우를 생각할 수 있습니다.

- **Raw Weighted Mean**

가중평균을 정말 단순하게 진행하는 것입니다

연관성 측도로 사용한 상관계수를 $Sim(u,v)$라고 정의했습니다(윗쪽 수식 참고). 이를 참고하여 가중평균을 Compute 하면 아래 수식으로 유도할 수 있습니다. 
$$
\hat r_{uj} = \frac{\sum_{v \in P_u (j)}Sim(u,v)* r_{vj}{}}{\sum_{v \in P_u (j)}Sim(u,v)}
$$
​	앞선 예제를 다시 살펴본다면, 

![image-20201231132651204](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231132651204.png)

​	여기서 k=2이고, 3번 유저의 1번 아이템을 추정한다고 하겠습니다. 3번 유저와 유사한 상위 두명의 유저는 1번 유저와 2번 유저입니다. 따라서 이들에 대해서 피어슨 상관계수를 그 연관성 척도로 사용한다면, 아래처럼 간단히 계산될 수 있습니다.


$$
\frac{7*0.894 + 6*0.939}{0.894+0.939} = \frac{11.892}{1.833} = 6.49
$$


​	이 방법에는 명확한 한계가 존재합니다. 바로 추정해야하는 유저의 특성이 전혀 반영되지 않았다는 점입니다. 3번 유저는 잘보시면 평균 평점이 2인 굉장히 평가가 박한 유저입니다. 근데 상대적으로 후한 점수를 주는 1번 2번 유저에 의해 굉장히 높게 추정되었네요. 이를 반영하기 위해 우리는 새로운 방법을 떠올릴 수 있습니다. 



- Centered Data Mean

  이 명칭은 정식 명칭은 아닙니다. 우리가 다른 유저로부터 반영할 정보는 '**타겟 유저와 유사한 다른 유저들은 이 아이템을 상대적으로 얼마나 더/덜 좋아했는가?**'입니다. 따라서 그 정보는 아래와 같이 편차로 대체할 수 있습니다.

$$
s_{uj} = r_{uj} - \mu_u  ~~~u \in \{1,2,3, ...,m\}
$$

​	앞선 가중평균과 거의 유사하지만, 이번에는 편차에 대한 가중평균을 내서, 그 유저의 평균에 더해주는 정도로만 계산합니다.

![image-20201231212818160](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231212818160.png)

​	편차에 대한 가중평균을 내고, 유저의 평균을 더해주면 되는 구성입니다. 구체적인 계산식은 아래와 같이 계산됩니다.

![image-20201231213905928](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231213905928.png)

​	3번 유저는 1번 아이템을 3.35, 6번 아이템을 0.86을 줄 것 같군요!



### 3. Reliability & Scaling

​	근데 우리가 간과한 한가지 중요한 사실이 있습니다. 바로 두 유저가 유사하긴한데, **'과연 그 유사성을 신뢰할 수 있는가?'**입니다. 그 신뢰성은 두 유저가 얼마나 많은 교집합을 갖느냐로 판단할 수 있습니다. 1개의 아이템만 겹치는 유저간 유사성과, 10개의 아이템이 겹치는 유저간 유사성은 분명 그 신뢰성이 다릅니다. 이러한 부분을 고려하여 다음과 같은 Discounted Similarity를 생각할 수 있겠습니다. 

![image-20210101180532762](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20210101180532762.png)

​	여기서 $\beta$는 Hyperparameter로 일종의 Threshold로 작용합니다. 대부분의 경우에 이 $\beta$는 1 이상의 정수값을 취합니다. 겹쳐지는 개수가 $\beta$보다 많으면 저 min()부분은 $\frac{\beta}{\beta} = 1$이 되어 아무런 Penalty를 주지 않지만, $\beta$보다 작은 값을 취하는 경우에는 0과 1사이의 값이 곱해져 유사도를 낮춥니다. 

​	근데 과연 평균을 빼는것만으로 Scaling이 될까요? Standard Scaling, MinMax 등등 다양한 표준적인 Scaling 기법을 보면 분모에 표준편차가 들어갑니다. 이것이 NN모델에도 들어갈 필요가 있을것으로 보입니다. 1~7점 중 다양하게 주는 유저에게 4점은 평균적인 평가일 것입니다. 그러나 4 5 6점만 주는 유저에게 4점은 정말 안좋은 점수입니다. 이를 적절하게 반영하기 위하여 앞서 진행한 Centered 작업에서 표준편차로 나눠주는 작업을 추가할 수 있습니다. 

![image-20201231215627817](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231215627817.png)      ![image-20201231215638031](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231215638031.png)![image-20201231215704263](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20201231215704263.png)   



### 4. Impact of the Long Tail

​	위에서 언급한 Long Tail 현상과 관련하여, 인기 아이템 중심의 추천현상은 직관적으로 당연한 현상입니다. 연관성을 추정하는 경우, 겹쳐지는 아이템은 대부분 major한 아이템들일테고, 그 아이템에 대한 두 유저의 성향만 연관성 측정에 영향을 줄 것 입니다. Prediction을 넘어서서 만약 그 예측된 Rating을 토대로 아이템을 내림차순 정렬하여, 상위 K개의 아이템을 추천한다고 했을때, 분명 유명한 아이템 중심으로 그 추천이 형성될 것입니다. 그러나 이는 추천시스템의 근본적 개발 이유와 부합하지 않습니다.

​	이를 보정하기 위해서, 상대적으로 많이 관측된 아이템에는 제약을 주고, 적게 관측된 아이템에 대해서는 가중을 주는 방식의 방법을 생각할 수 있겠습니다. 

![image-20210101173500920](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20210101173500920.png)

​	이러한 Weight를 생각해보겠습니다. 여기서 m은 유저의 수이고, $m_j$는 m명의 유저 중 j번째 아이템에 rating을 매긴 유저입니다. 빈번하게 등장하는 아이템의 경우에는 저 분모값이 매우 커지기 때문에 상대적으로 적은 가중치를 줄 것이고, 빈도수가 적은 아이템의 경우에는 저 가중치값이 매우 커질 것입니다. 이 가중치를 고려해서 다시 유저간 관계척도를 보정하면 다음과 같습니다. 

![image-20210101174121730](C:\Users\kswoo\AppData\Roaming\Typora\typora-user-images\image-20210101174121730.png)

​	이렇게 weight인 $w_k$를 곱한 형태를 구할 수 있겠습니다.

​	

​	오늘은 여기서 마무리를 하고, 다음에는 이어서 Item Based Collaborative Filtering에 대해서 알아보겠습니다.

