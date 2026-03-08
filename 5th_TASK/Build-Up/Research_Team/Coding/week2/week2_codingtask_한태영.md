# RNN Encoder-Decoder 핵심 코드 (Cho et al., 2014)

> 독일어 문장 → context vector → 영어 문장 한 토큰씩 생성

---

## 참고 및 출처

- **논문**: Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- **참고 코드**: [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) : 2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

---

## 1. Encoder: 소스 문장 → context vector

**역할**: 입력 시퀀스(독일어) 전체를 읽고, 마지막 hidden state를 "문장의 의미"로 사용

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # 토큰 ID → 벡터   Input_dim = vocab 크기
        self.rnn = nn.GRU(embedding_dim, hidden_dim)             # LSTM 대신 GRU (구조 단순)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [seq_len, batch] → 토큰 ID 시퀀스
        embedded = self.dropout(self.embedding(src))   # [seq_len, batch, emb_dim] [문장 토큰 개수, 한 번에 처리하는 문장 수, 벡터길이]
        outputs, hidden = self.rnn(embedded)           # hidden: [1, batch, hidden_dim]
        return hidden   # ← 이게 context vector (디코더에 전달되어 디코더의 context 로 사용
```

---

## 2. Decoder: context + 이전 토큰 → 다음 토큰 예측

**역할**: 매 스텝마다 (1) 이전 토큰 (2) context 를 함께 보고, 다음 영어 토큰의 확률 분포 출력

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout):   
        # [output_dim	, embedding_dim	, hidden_dim] [출력 vocab 크기 (영어), 토큰 ID → 벡터 길이, RNN hidden/context 크기	]
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        # 입력 = [현재 토큰 embedding + context] → GRU가 "원본 문장" 정보를 매 스텝 참조
        # 영어 토큰 ID(예시) → 256차원 벡터
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim)
        # 입력 크기(예시) = 256 + 512 = 768
        # 이유: [이전 토큰 embedding] + [context] 를 합쳐서 넣음
        # 출력 hidden = 512 (인코더 hidden과 동일)
        self.fc_out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        # 입력 = 256 + 512*2 = 1280
        # (embedding 256) + (GRU hidden 512) + (context 512)
        # 출력 = 5893 (vocab 크기)
    def forward(self, input, hidden, context):
        # input: [batch] = 이번에 넣을 토큰 ID
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))  # [1, batch, emb_dim]
        emb_con = torch.cat((embedded, context), dim=2)              # context concat
        output, hidden = self.rnn(emb_con, hidden)
        concat = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(concat)  # [batch, vocab_size]
        return prediction, hidden
```

---

## 3. Seq2Seq: Encoder + Decoder 연결

**역할**: Encoder로 context 추출 → Decoder가 `<sos>`부터 한 토큰씩 생성, `<eos>`까지

```python
class Seq2Seq(nn.Module):
    def forward(self, src, trg, teacher_forcing_ratio):
        # src: 독일어 [seq_len, batch], trg: 영어 [seq_len, batch]
        context = self.encoder(src)   # ① 소스 전체 → context
        hidden = context             # 디코더 초기 hidden = context

        outputs = torch.zeros(...)
        input = trg[0, :]            # ② 첫 입력 = <sos>

        for t in range(1, trg_length):
            output, hidden = self.decoder(input, hidden, context)  # ③ 다음 토큰 예측
            outputs[t] = output

            # Teacher Forcing: 학습 시 정답을 넣을지, 예측을 넣을지 확률적으로
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output.argmax(1)

        return outputs   # [seq_len, batch, vocab_size]
```

---

## 4. 학습 시 손실 계산

```python
# output[1:]: <sos> 제외 (입력이었으므로 예측 대상 아님)
# trg[1:]: 정답도 <sos> 제외
output = output[1:].view(-1, vocab_size)
trg = trg[1:].view(-1)
loss = CrossEntropyLoss(ignore_index=pad_index)(output, trg)
```

---

## 5. 추론 시 흐름

```python
# ① 독일어 → context
context = encoder(de_ids)

# ② <sos>부터 시작해 한 토큰씩 생성
input = [sos_id]
for _ in range(max_len):
    output, hidden = decoder(input[-1], hidden, context)
    next_token = output.argmax(-1)
    input.append(next_token)
    if next_token == eos_id:
        break
```

---

## 요약

| 구성 요소 | 입력 | 출력 |
|----------|------|------|
| Encoder | src (독일어 토큰 ID) | context (고정 벡터) |
| Decoder | 이전 토큰 + hidden + context | 다음 토큰 확률 분포 |
| Seq2Seq | src, trg (학습용) | 전체 시퀀스에 대한 예측 |

---





### 1. 헷갈렸던 점 + 정리

- **질문**: SOS? EOS? 
- **정리**: SOS = Start of Sequence "디코더 첫 번째 입력" . EOS = End of Sequence "이 토큰 나오면 디코더가 멈춤"


### 2. 요약

- Encoder: 고정벡터로 출력
- Decoder: 현재 토큰 embedding + context (원본문장과 Encoder 가 출력한 C 활용 예측)
- Seq2Seq: Encoder와 Decoder로 구성된 번역기
