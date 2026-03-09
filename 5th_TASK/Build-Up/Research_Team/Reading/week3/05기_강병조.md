**Week3 Reading Task**

**📘 Title**

> ℹ️ Neural Collaborative Filtering
> 

---

**📖 Abstract**

> ℹ️ 본인의 방식으로 재해석 해주세요. 그대로 가져오는 것은 금합니다.
> 본 논문에서는 추천 시스템에 딥러닝을 접목시켜 Neural Collaborative Filtering이라는 새로운 모델을 제안한다. 추천 시스템 모델에서 가장 대표적으로 사용되는 Matrix Factorization의 한계를 보완하고자 일반화된 Matrix Factorization 모델을 제안하고, 여기에 다층 퍼셉트론(MLP) 구조를 결합하여 GMF, MLP, NeuMF 총 세 가지 모델을 비교 분석한다. 



---

**📚 Background**

> ℹ️ 논문의 주제와 관련된 기존 연구들 및 배경 지식을 정리
> 
> 
 **📍 Related Work 1: A generic
coordinate descent framework for learning from implicit
feedback (I. Bayer et al., 2017)**
>
기존 Explicit Feedback을 이용하던 추세에서 조회수, 조회 기록 등 implicit data를 사용하는 방식 도입 및 성능 시사
> 
 **📍 Related Work 2: Autorec:
Autoencoders meet collaborative filtering (S. Sedhain et al., 2015)**
> 
Matrix Factorization의 한계를 보완하기 위해 autoencoder 구조 제안하기 시작함. 이 논문에서는 AutoRec를 제안하여 과거 별점 기록을 input으로 사용하는 방식 제안
> 

---

**🔍 Methods**

✅ **사용된 연구 방법**: 
> Matrix Factorization의 한계를 보완하기 위한 프레임워크 제안
>
> Input Layer - Embedding Layer - Neural CF layers - Output Layer
>
> - Embedding Layer에서는 Input 단계의 sparse한 vector들을 dense하게 변화시키는 작업을 수행한다.
> - Neural CF Layers에서는 mapping된 latent vector들을 이용하여 인공신경망 모델을 통해 점수화시킨다.
> 
 **📍 모델 비교**: 
> - GMF
> - MLP
> - NeuMF(GMF + MLP)

---

**🔍 Experiments**

 ✅ **데이터셋**: 
> 1. MovieLens 데이터셋 (1,000,209 Interactions, 3,706 Items, 6,040 Users)
> 2. Pinterest 데이터셋 (1,500,809 Interactions, 9,916 Items, 55,187 Users)
> 
**✅ Models:** 
> - 논문에서 제안한 GMF, MLP, NeuMF
> - 기존 모델 ItemKNN, ItemPop, BPR, eALS
> 
**✅ Evaluation Metrics:** 
> - Leave-one-out Evaluation 사용
> 
> 각 User에 대해 마지막으로 Interaction을 가진 item과 이 User와 Interaction이 없는 100개의 item을 임의추출
> 
> 101개의 item에 대해 rank를 매겨 마지막으로 Interaction을 가진 item의 rank를 HR(Hit Ratio)와 NDCG(Normalized Discounted Cumulative Gain)로 확인
> 
**✅ Implementation Details:** 
> - HR과 NDCG로 모델 성능 비교
> - Training Loss 확인 후 비교
> - MLP layer의 수를 다르게 하여 성능 비교

---

**📖 Conclusion**

 ✅ **Limitation:** 
> - Sparse한 데이터셋에 대해서만 실험하여 dense한 경우에 대한 결과를 밝히지 않음
> 
 **✅ Contribution:** 
> - 기존에 없던 새로운 딥러닝 기반 프레임워크 제안
> - Deep Neural Network가 성능 개선이 도움이 되는 것을 MLP layer의 수를 다르게 실험함으로써 확인 (Section 4.4)
> 

---

**🤔 Question**

> ℹ️ 본인이 수행한 학습에 대해 스스로 질문하고 답해보세요.
> 
> **Q1. 논문에서 모델의 성능을 측정하기 위해 사용된 HR과 NDCG는 무엇인가?**
>
> A. HR@K는 K개의 상품을 추천하라고 하였을 때 TRUE 값이 있는 경우를 1, 없는 경우를 0으로 하여 비율을 구한 값이다.
> NDCG@K는 마찬가지로 K개의 상품을 추천하라고 하였을 때 높은 등수에 TRUE 값이 있는 경우에 가중치를 주어 구한 값이다.
> 

> **Q2. Implicit Feedback과 Explicit Feedback이 의미하는 바는 무엇인가?**
>
> A. Explicit Feedback은 별점과 같이 직접적으로 선호도를 표시하는 척도이고, Implicit Feedback은 조회수, 즐겨찾기 여부 등과 같이 간접적으로 선호도를 표시하는 척도이다. 따라서 논문에서도 언급된 내용인 Implicit Feedback의 경우 이진분류 하였을 때 0이 싫어한다는 의미가 아닌, 상호작용이 없음을 의미한다.

