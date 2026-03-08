# week2 Coding Task

## 목표
- Encoder/Decoder 구조 이해
- forward 흐름 정리
- 추론 시 한 토큰씩 생성 방식 이해
- GRU가 코드에서 어떻게 구현되는지 확인

---

## 핵심 모델 구조
### Encoder
Input → LSTM(return_state=True) → encoder_states  
- 입력 시퀀스를 받아 마지막 hidden state를 반환
- 이 hidden state가 문장 전체를 요약한 context 벡터 역할

### Decoder
Input → LSTM(initial_state=encoder_states) → Dense(softmax)
- Encoder의 마지막 상태를 초기 상태로 사용
- 각 시점(time step)마다 다음 토큰의 확률을 예측
- Dense + softmax를 통해 단어 분포 출력

---

## Forward 흐름
1. 입력 시퀀스를 Encoder에 넣어 `encoder_states` 생성  
2. Decoder는 `encoder_states`를 초기 상태로 받아 출력 생성  
3. 학습 단계에서는 teacher forcing 사용  
4. 추론 단계에서는 while 루프로 한 토큰씩 생성

### 학습 VS 추론
- 학습 단계: teacher forcing 사용 → 이전 정답 토큰을 다음 입력으로 사용
- 추론 단계: while 루프 사용 → 이전에 예측한 토큰을 다음 입력으로 사용 → 한 토큰씩 순차적으로 생성

---

## 학습 모델 코드
```python
from keras.models import Model
from keras.layers import Input, LSTM, Dense

encoder_inputs = Input(shape=(None, num_encoder_tokens))
# (batch_size, time_step, vocab_size)
# 가변 길이 입력 시퀀스

encoder = LSTM(latent_dim, return_state=True)
# return_state=True → 마지막 hidden state와 cell state 반환

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]
# 문장을 요약한 context 벡터 역할

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Decoder 입력 (teacher forcing 시 정답 문장)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# return_sequences=True → 모든 시점의 출력 필요
# return_state=True → 다음 단계 state 전달용

decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs,
    initial_state=encoder_states
)
# Encoder의 마지막 state를 초기 상태로 사용

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# 각 시점마다 다음 토큰의 확률 분포 계산

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```
#### 핵심
- initial_state=encoder_states가 Encoder–Decoder 연결 포인트
- return_state, return_sequences가 왜 필요한지
- softmax가 확률분포 출력

---

## 추론 코드
```python
def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)
    # 입력 문장을 Encoder에 넣어 context(state) 생성

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    # 시작 토큰 <start> 입력

    stop_condition = False
    decoded_sentence = ''


    # 이전에 예측한 토큰을 다음 입력으로 사용하며 한 토큰씩 생성
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
        )
        # 현재 입력과 이전 state를 이용해 다음 토큰 예측

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # 가장 확률이 높은 토큰 선택

        sampled_char = reverse_target_char_index[sampled_token_index]

        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
            # '\n'을 <end> 토큰처럼 사용해서 문장 생성 종료

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # '\t'을 <start> 토큰처럼 사용 (데이터 전처리에서 시작 토큰으로 지정된 경우)

        states_value = [h, c]
        # 다음 시점에서 사용할 state로 업데이트

    return decoded_sentence
```

---

## GRU 버전
Cho et al. (2014) 논문에서 제안한 Encoder–Decoder 구조는 GRU 기반 RNN이다.
GRU는 reset gate와 update gate를 통해 정보를 얼마나 유지하고 버릴지 결정하는 구조이다.

```python
from keras.layers import GRU

encoder = GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)
# GRU는 hidden state 하나만 반환 (cell state 없음)

decoder_gru = GRU(latent_dim, return_sequences=True)
decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
# Encoder의 hidden state를 Decoder 초기 상태로 사용
```

---

## 정리
- Encoder의 마지막 상태가 문장 요약 역할
- Decoder는 이를 초기 상태로 사용
- 학습은 teacher forcing
- 추론은 while 루프
- GRU는 gate가 포함된 RNN 유닛
