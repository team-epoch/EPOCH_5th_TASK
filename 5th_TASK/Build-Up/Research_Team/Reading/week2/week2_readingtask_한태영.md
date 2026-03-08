**Week0 Reading Task**

**📘 Title**

> ℹ️ (논문 제목)
> Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
Cho et al., EMNLP 2014
---

**📖 Abstract**

> 

과거의 번역 방식(SMT)에서 새로운 번역 방식(NMT)으로 나아가게 해준 RNN Encoder-Decoder. 논문의 RNN방식을 통해 문장을 고정된 크기의 벡터로 압축하여 처리해 가변 길이의 문제를 해결하였으며, GRU(게이트 순환 유닛)가 탄생하게 되는 배경이 되었다. 작동방식은 인코더가 정보를 압축하고, 디코더가 그 압축본을 바탕으로 새로운 문장을 생성해낸다는 것이다.여기서 주의 깊게 봐야 할 점은 언어의 규칙성 또한 학습할 수 있게 된 점이다. 실효성은 기존 기계 번역 시스템의 구문 채점에 적용하여 번역 성능이 향상됨을 통해 보여주었다. 

---

**📚 Background**

 ℹ️ 논문의 주제와 관련된 기존 연구들 및 배경 지식을 정리
> 
> 
 **📍 Related Work 1:(Zou et al., 2013) proposed to learn a bilingual embedding of words/phrases**
> 
Zou팀은 영어/불어 단어들의 이중언어 임베딩을 학습시킴. 단어의 뜻을 숫자로 바꾼다는 아이디어는 좋았지만, 문장을 통째로 읽고 뱉어 내는 완벽한 신경망 구조(End-to-End)에는 도달하지 못함.
> 
 **📍 Related Work 2:In (Chandar et al., 2014), a feedforward neural network**
> 
Chandar 연구팀은 feedforward 신경망을 사용해 입출력 구문을 학습시킴.입력 문장을 Bag-of-words방식으로 넣는다는 단점이 존재했다. 
어순을 고려하지 못해서 
1.I am hantaeyoung 
2.hantaeyoung am i 
두 문장을 똑같은 의미로 인식하는 한계가 있었다
> 

---

**🔍 Methods**

✅ **사용된 연구 방법**: 

RNN 방식: 다음에 올 단어를 예측하도록 훈련 받음으로 시퀀스의 확률분포 학습 가능

RNN Encoder-Decoder: encoder와 decoder로 구성되어있음. encoder가 문장을 다 읽고 나서 c(문맥벡터)로 요약을 하고, decoder가 지난 단어 $y_{t-1}$ 와 c의 영향을 받아서 $y_{t}$ 와 $h_{t}$ 를 갱신함.

식:
$$h_{\langle t \rangle} = f(h_{\langle t-1 \rangle}, y_{t-1}, \mathbf{c})$$

encoder와 decoder 는 조건부 로그 우도를 최대화하도록 묶여서 훈련된다

$$\max_{\boldsymbol{\theta}} \frac{1}{N} \sum_{n=1}^{N} \log p_{\boldsymbol{\theta}}(\mathbf{y}_n \mid \mathbf{x}_n)$$

**[기호 설명]**

- $N$: 전체 학습 데이터(문장 쌍)의 총 개수
- $\mathbf{x}_n$: $n$번째 입력 문장 (원본 영어 문장)
- $\mathbf{y}_n$: $n$번째 타겟 문장 (정답 프랑스어 번역)
- $p_{\boldsymbol{\theta}}(\mathbf{y}_n \mid \mathbf{x}_n)$: 모델 파라미터 $\boldsymbol{\theta}$가 주어졌을 때, $\mathbf{x}_n$을 보고 $\mathbf{y}_n$을 맞출 확률
- $\max_{\boldsymbol{\theta}}$: 정답을 출력할 확률을 최대화하도록 모델을 학습시킴

reset gate 와 update gate 를 통해 새로운 기억을 업데이트 나감.

reset gate: r의 값이 0에 가까워지면, 인공지능은 과거의 기억을 무시하고 오직 현재 들어온 단어에만 집중
Ex. I am taeyoung, but... 'but 이라는 단어를 보고 과거의 문맥을 끊고 새로운 단어 준비

update gate: 과거의 기억 $h_{t-1}$을 얼마나 가지고 올지 결정함.

각 hidden unit 마다 resetgate 와 update 게이트가 따로 달려있어 서로 다른 시간대의 문맥을 잡는 법을 학습함. 단기기억과 장기기억의 구성이 다름
> 
 **✅ 실험 설계:** 

사용 데이터셋: English/French translation task of the WMT'14workshop 

데이터 필터링 후 
1. Baseline configuration
2. Baseline + RNN
3. Baseline + CSLM + RNN
4. Baseline + CSLM + RNN + Word penalty

4개 모델의 Development / Test set에 대한 BLEU 점수 비교

 **📍 모델 비교**: 
| Models | dev (BLEU) | test (BLEU) | 비고 |
| :--- | :---: | :---: | :--- |
| **Baseline** | 30.64 | 33.30 | 기존 시스템 |
| **RNN** | 31.20 | 33.87 | 성능 향상 |
| **CSLM + RNN** | 31.48 | **34.64** | **최고 성능** |
| **CSLM + RNN + WP** | 31.50 | 34.54 | WP 추가 |
 
 WP (Word Penalty): 번역 과정에서 생성되는 단어 수에 패널티를 주어 길이를 조절하는 옵션

**🔍 Experiments**

 ✅ **데이터셋**: 
 
 English/French translation task of the WMT'14 workshop 데이터셋의 노이즈 처리, 필터링 함.

**✅ Models:** 
 1. Baseline configuration (기존 시스템)
2. Baseline + RNN     (기존시스템 + RNN(채점에 사용)    )
3. Baseline + CSLM + RNN (기존시스템 + CSLM + RNN)
4. Baseline + CSLM + RNN + Word penalty (기존시스템 + CSLM + RNN + 패널티 옵션 추가)

**✅ Evaluation Metrics:** 
 
 BLEU score 사용 : 사람이 직접 번역한 결과와 얼마나 유사한지 나타내는 지표
4개 모델의 Development / Test set에 대한 BLEU 점수 비교

**✅ Implementation Details:** 


Rank-100 matrices: 행렬 연산의 효율성을 위해 Rank-100 행렬 사용

Hyperbolic tangent (tanh): 은닉층의 활성화 함수(Activation function)로 사용

Maxout layer: 디코더에서 최종 출력으로 이어질 때 Maxout 신경망 계층을 거치도록 설계

Gaussian distribution: 순환 가중치(Recurrent weights)를 제외한 모든 가중치 파라미터를 정규분포를 따르도록 초기화(Initialization)

Optimization: Adadelta 및 확률적 경사 하강법(SGD, Stochastic Gradient Descent)을 사용하여 모델 최적화

**📖 Conclusion**

 ✅ **Limitation:** 
 1. 부분 대체 역할: 기존 번역기 전체를 100% 딥러닝으로 교체한게 아닌 기존 방식인 SMT의 구문 채점 역할로 검증했다는 한계가 있다.
>
2. 고정벡터의 한계: 가변 길이 문장을 단 하나의 '고정 길이 벡터'로 압축해야 하는 구조적 한계 존재 

     **긴문장에서 손실이 발생하며 향후 Attention 메커니즘이 등장하는 배경이 됨**  
> 
 **✅ Contribution:** 
 1. RNN Encoder-Decoder 구조 확립:
 입력과 출력의 길이가 달라도 처리할 수 있는 새로운 신경망 구조를 제안하여, 향후 NMT(신경망 기계 번역) 및 Seq2Seq 모델의 핵심 뼈대를 마련했다.

 2. 새로운 순환 유닛(GRU)의 탄생:
 기존 RNN의 기울기 소실 및 장기 기억 상실 문제를 극복하기 위해, 연산이 효율적이면서도 기억을 적응적으로 조절하는 리셋/업데이트 게이트(Reset & Update Gate) 구조를 최초로 제안했다.

 3. 언어의 의미/문법적 규칙성 증명:
 기계가 단순히 데이터의 빈도수만 외우는 것이 아니라, 단어와 구문의 의미적(Semantic) 및 문법적(Syntactic) 구조를 스스로 이해하고 다차원 연속 공간에 맵핑할 수 있음을 완벽하게 입증해 냈다

---

**🤔 Question**

 
**Q1.저자가 논문의 RNN Encoder-Decoder 방식을 떠올리게 된 계기가 뭘까?**

A. 기존의 방식은 입력되는 문장의 길이가 고정되어 있어야 하거나, 어순을 무시하는 방식을 사용했었음. 하지만, 인간의 언어는 길이가 제각각이고 어순이 중요했다. 이를 해결하기 위해 순서대로 읽는 데 특화된 RNN을 두개로 쪼개 분업화하는 방식을 떠올림.

**Q2.몇 년이 지난 지금(2026년), 문학 작품 번역에서 작가의 숨은 의도를 인간 번역가처럼 완벽하게 전달하지 못하는 이유는?**

A. <span style="color:red">기계가 학습한 것은 방대한 데이터 속의 통계적인 패턴과 언어의 문법/의미적 규칙성 일뿐, 인간의 경험과 문화가 아니기 때문.</span> 아직 인간이 AI와 차별점이 있는 부분

**Q3.논문에서 제안한 모델을 훈련시키기 위해, 어느 정도의 시간과 컴퓨팅 자원이 소요되었을까?**

A. 당시의 하드웨어 기준으론 최소 며칠에서 몇 주가 걸렸을 것으로 예상. RNN의 순차적 연산의 특성 때문에 병렬 처리가 어려워 학습속도가 느림. (속도 문제 해결을 위해 향후 Transformer 모델이 등장)

**Q4.이 논문의 영향으로 현재(2026년)우리는 어떤 혜택을 누리고 있는가?**

A. 신경망 기계 번역 시대의 뼈대(Seq2Seq)를 세움. 논문에서 최초로 고안된 게이트 구조는 AI의 기억력을 향상시킴. LLM이 문맥 이해의 근간이 됨.

