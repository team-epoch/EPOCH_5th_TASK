**Week2 Coding Task**

> 코드 해석 중심으로 공부해봤습니다.
> 
> 여기 링크도 참조하였습니다! https://happy-jihye.github.io/nlp/nlp-7/
---

 **Encoder**: 

```
class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, dropout):
    super().__init__()

    self.hid_dim = hid_dim

    self.embedding = nn.Embedding(input_dim, emb_dim)

    self.rnn = nn.GRU(emb_dim, hid_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, src):

    # src = [src len, batch size]
    embedded = self.dropout(self.embedding(src))

    # embedded = [src len, batch size, emb dim]

    ## cell state가 없습니다 !
    outputs, hidden = self.rnn(embedded)

    # outputs = [src len, batch size, hid dim * n directions]
    # hidden = [n layers * n directions, batch size, hid dim]

    ## output은 언제나 hidden layer의 top에 있습니다.

    return hidden
```

> input으로 직전 단계 (self)만 사용

 **Decoder**: 
```
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # input : context vec + embedding vec
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, context):
        
        # input = [batch size]
        ## 한번에 하나의 token만 decoding하므로 forward에서의 input token의 길이는 1입니다.
        
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        # input을 0차원에 대해 unsqueeze해서 1의 sentence length dimension을 추가합니다.
        # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        # embedding layer를 통과한 후에 dropout을 합니다.
        # embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context), dim = 2)
        
        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim = 1)
        
        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden
```

> encoder와는 다르게 input으로 hidden layer + context vector 사용

 **최종 Seq2Seq**: 
```
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # output을 저장할 tensor를 만듭니다.(처음에는 전부 0으로)
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # src문장을 encoder에 넣은 후 context vector를 구합니다.
        context = self.encoder(src)
        
        # decoder의 initial hidden state는 context vector입니다.
        hidden = context

        # decoder에 입력할 첫번째 input입니다.
        # 첫번째 input은 모두 <sos> token입니다.
        # trg[0,:].shape = BATCH_SIZE 
        input = trg[0,:]  
        
        
        '''한번에 batch_size만큼의 token들을 독립적으로 계산
        즉, 총 trg_len번의 for문이 돌아가며 이 for문이 다 돌아가야지만 하나의 문장이 decoding됨
        또한, 1번의 for문당 128개의 문장의 각 token들이 다같이 decoding되는 것'''
        for t in range(1, trg_len):
            
            # input token embedding과 이전 hidden state와 context state를 decoder에 입력합니다.
            # 새로운 hidden state와 예측 output값이 출력됩니다.
            output, hidden = self.decoder(input, hidden, context)

            #output = [batch size, output dim]

            # 각각의 출력값을 outputs tensor에 저장합니다.
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # predictions들 중에 가장 잘 예측된 token을 top에 넣습니다.
            # 1차원 중 가장 큰 값만을 top1에 저장하므로 1차원은 사라집니다.
            top1 = output.argmax(1) 
            # top1 = [batch size]
            
            # teacher forcing기법을 사용한다면, 다음 input으로 target을 입력하고
            # 아니라면 이전 state의 예측된 출력값을 다음 input으로 사용합니다.
            input = trg[t] if teacher_force else top1
        
        return outputs
```




