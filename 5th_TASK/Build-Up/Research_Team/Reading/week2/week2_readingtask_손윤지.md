**Week0 Reading Task**

**📘 Title**

> ℹ️ **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation**
> 
> 
> (Cho et al., 2014)
>
> 

---

**📖 Abstract**

> ℹ️ 본인의 방식으로 재해석 해주세요. 그대로 가져오는 것은 금합니다.
> 

이 논문은 통계적 기계번역 시스템에서 사용되는 phrase 단위 번역 확률을 더 효과적으로 모델링하기 위해, 새로운 신경망 구조인 **RNN Encoder–Decoder**를 제안하는 논문이었다. 본 모델은 입력 phrase를 하나의 연속적인 벡터 표현으로 압축한 뒤, 이를 기반으로 대응되는 출력 구를 생성하는 구조를 가진다. 특히 기존 RNN의 장기 의존성 문제를 완화하기 위해 **gating mechanism을 포함한 새로운 recurrent unit(GRU)** 을 도입하였다.

실험 결과, 제안한 모델은 기존 SMT 시스템에 추가적인 확률 점수로 활용될 때 번역 성능을 향상시켰다. 또한 학습된 벡터 표현은 의미적·문법적으로 유사한 구들을 가까운 위치에 배치하는 특성을 보였다. 이는 신경망이 구의 구조적 및 의미적 정보를 효과적으로 학습했음을 보여준다.

---

**📚 Background**

> ℹ️ 논문의 주제와 관련된 기존 연구들 및 배경 지식을 정리
> 
> 
 **📍 Related Work 1: Neural Language Models (Bengio et al., 2003)**
> 
> 연속적인 벡터 공간에서 단어를 표현하는 아이디어를 제시하였다. 이는 희소 표현 문제를 완화하고 의미적 유사성을 반영할 수 있는 기반을 마련했다.
> 
 **📍 Related Work 2: Recurrent Neural Networks & LSTM**
>
> 기존 RNN과 LSTM은 시퀀스 데이터를 모델링하는 데 사용되었지만, 구조가 복잡하고 학습이 어렵다는 문제가 있었다. 본 논문은 LSTM보다 단순한 구조의 GRU를 제안하여 이를 개선하였다.


---

**🔍 Methods**

✅ **사용된 연구 방법**: 
- 새롭게 고안된 RNN Encoder–Decoder 구조 제안
- 입력 시퀀스를 고정 길이 벡터로 압축
- 해당 벡터를 기반으로 출력 시퀀스 생성
- 새로운 recurrent unit인 GRU 설계

 ✅ **실험 설계:** 
- phrase pair를 입력-출력 쌍으로 구성
- 조건부 확률 ( p(y|x) ) 최대화
- 학습된 확률을 기존 SMT 시스템의 feature로 추가
- BLEU score로 성능 평가
  
 **📍 모델 비교**: 
- 기존 SMT phrase probability
- RNN Encoder–Decoder 기반 확률 점수
- log-linear model에 통합 후 성능 비교

---

**🔍 Experiments**

 ✅ **데이터셋**: 
- WMT’14 English–French 병렬 코퍼스
- 기존 phrase table에서 추출된 phrase pair
  
**✅ Models:** 
- 기존 통계적 번역 모델
- 제안한 RNN Encoder–Decoder (GRU 기반)

**✅ Evaluation Metrics:** 
- BLEU score
- 번역 품질 비교
> 
**✅ Implementation Details:** 
- 확률 최대화 기반 학습
- minibatch SGD 사용
- phrase-level training
- fixed-length context vector 사용 

---

**📖 Conclusion**

 ✅ **Limitation:** 
- 입력 문장을 하나의 고정 길이 벡터로 압축하는 구조는 긴 문장에서 정보 손실을 초래함
- 완전한 end-to-end 신경망 번역 시스템은 아님
- attention mechanism 부재
> 
 **✅ Contribution:**
1. Encoder–Decoder 구조의 최초 제안
2. GRU(Gated Recurrent Unit) 도입
3. phrase-level neural scoring을 SMT에 성공적으로 적용
4. Neural Machine Translation 연구의 기반 마련
5. 이후 Attention 모델 등장의 큰 기여점이 됨 

---

**🤔 Question**

> ℹ️ 본인이 수행한 학습에 대해 스스로 질문하고 답해보세요.
> 
Q1. 왜 고정 길이 벡터가 번역 성능을 제한하는가?

A1. 가변 길이 입력을 단일 벡터로 압축하는 과정에서 정보 병목이 발생한다. 문장이 길어질수록 모든 정보를 하나의 벡터에 담아야 하므로 표현 용량이 제한된다. 이는 긴 문장에서 성능 저하로 이어진다. 본 문제는 모델의 학습이 되지 않은 시점에서도 발견되었다는 특징이 있었다. 이를 해소하기 위한 모델이 'Attention All You Need'이라는 논문에서 추후 제안된 Transformer로 해소된다.

Q2. GRU는 왜 LSTM보다 단순하면서도 효과적인가?

A2. GRU는 cell state와 hidden state를 분리하지 않고, 두 개의 gate만을 사용하기 때문! update gate가 과거 정보를 유지하는 경로를 제공하여 gradient 흐름을 개선하며, 구조가 단순해 학습이 빠르다.

