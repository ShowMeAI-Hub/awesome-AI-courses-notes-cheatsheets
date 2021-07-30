# Assignment5讲解

Assignment5的任务相对Assignment4要更进一步，虽然它的主题也是基于(字符级别的卷积)神经网络机器翻译的应用，但要完成的代码更多，官方给的API和代码实现提示也更少。

- [实验指导文档下载地址](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5.pdf)
- [代码与数据下载地址](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5_public.zip)

## 本次作业的官方说明
这个作业探索了两个关键概念——子词建模和卷积网络——并将它们应用于我们在上一个作业中构建的 NMT 系统。 Assignment 4 NMT 模型可以被认为是四个阶段：

- 1.嵌入层：通过查找将原始输入文本（对于源句子和目标句子）转换为密集词向量序列。
- 2.编码器：对源句子进行编码的 RNN作为编码器隐藏状态序列。
- 3.解码器：对目标句子进行操作并关注编码器隐藏状态以产生解码器隐藏状态序列的 RNN。
- 4.输出预测层：具有 softmax 的线性层，产生概率每个解码器时间步长上下一个目标词的分布。
所有这四个子部分都在词级别对 NMT 问题进行建模。

在本任务的第 1 节中，我们将用基于字符的卷积编码器替换 (1)，在第 2 节中，我们将通过添加基于字符的 LSTM 解码器来增强 (4)。这有望提高我们在测试集上的 BLEU 性能！最后，在第 3 节中，我们将检查我们的字符级编码器生成的词嵌入，并分析我们新 NMT 系统中的一些错误。

## part 1. 可用于神经网络翻译系统的字符级别卷积网络编码器
### (a)
答：因为字母的复杂度比单词要小很多，词表也更小，因此embedding大小也可以相应减小。

### (b)
答：
- character-based
    - $$
      V_{\text {char}} \times e_{\mathrm{char}}  + e_{\text{word}} \times e_{\mathrm{char}} \times k+e_{\text{word}}
      \approx 96\times 50+256\times 50\times 5+256=69056
      $$
- word-based
    - $$
      V_{\text {word}} \times e_{\text {word}} \approx 50000 \times 256 =12800000
      $$

### (c)

答：1D卷积的计算速度快（可并行），RNN计算速度慢（串行），而且卷积操作可以捕捉局部相关性。

### (d)

答：最大池化可以过滤无用信息，平均池化可以保留大部分信息。

### (e)

实现 vocab.py 中的words2charindices()函数，将每个字符转换为字符词汇表中对应的索引。

```python
def words2charindices(self, sents):
    """ Convert list of sentences of words into list of list of list of character indices.
    @param sents (list[list[str] ]): sentence(s) in words
    @return word_ids (list[list[list[int] ]]): sentence(s) in indices
    """
    ### YOUR CODE HERE for part 1e
    ### TODO: 
    ###     This method should convert characters in the input sentences into their 
    ###     corresponding character indices using the character vocabulary char2id 
    ###     defined above.
    ###
    ###     You must prepend each word with the `start_of_word` character and append 
    ###     with the `end_of_word` character. 
    res = []
    for sent in sents:
        l1 = []
        for word in sent:
            l2 = []
            l2.append(self.char2id['{'])
            for char in word:
                l2.append(self.char2id[char])
            l2.append(self.char2id['}'])
            l1.append(l2)
        res.append(l1)
    return res
    
    ### END YOUR CODE
```

使用如下代码测试：

```shell
python sanity_check.py 1e
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 1e: words2charindices()
--------------------------------------------------------------------------------
Running test on small list of sentences
Running test on large list of sentences
All Sanity Checks Passed for Question 1e: words2charindices()!
--------------------------------------------------------------------------------
```

### (f)

完成utils.py中的pad_sents_char()函数，使得其可以完成字符和词级别的padding

```python
def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int] ]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int] ]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents()
    ###     method below using the padding character from the arguments. You should ensure all
    ###     sentences have the same number of words and each word has the same number of
    ###     characters.
    ###     Set padding words to a `max_word_length` sized vector of padding characters.
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles
    ###     padding and unknown words.
    L = 0
    for sent in sents:
        L = max(L, len(sent))
        
    sents_padded = []
    for sent in sents:
        l1 = []
        for word in sent:
            l2 = []
            n = len(word)
            m = min(n, max_word_length)
            for index in word[:m]:
                l2.append(index)
            for i in range(max_word_length - m):
                l2.append(char_pad_token)
            l1.append(l2)
        #补充单词
        for i in range(L - len(sent)):
            l1.append([char_pad_token for j in range(max_word_length)])
        sents_padded.append(l1)

    ### END YOUR CODE

    return sents_padded
```

使用如下代码测试：

```
python sanity_check.py 1f
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 1f: Padding
--------------------------------------------------------------------------------
Running test on a list of sentences
Sanity Check Passed for Question 1f: Padding!
--------------------------------------------------------------------------------
```

### (g)

完成vocab.py文件中的to_input_tensor函数

```python
def to_input_tensor_char(self, sents: List[List[str] ], device: torch.device) -> torch.Tensor:
    """ Convert list of sentences (words) into tensor with necessary padding for 
    shorter sentences.

    @param sents (List[List[str] ]): list of sentences (words)
    @param device: device on which to load the tensor, i.e. CPU or GPU

    @returns sents_var: tensor of (max_sentence_length, batch_size, max_word_length)
    """
    ### YOUR CODE HERE for part 1g
    ### TODO: 
    ###     Connect `words2charindices()` and `pad_sents_char()` which you've defined in 
    ###     previous parts
    charindices = self.words2charindices(sents)
    charindices_pad = pad_sents_char(charindices, self.char2id['<pad>'])
    res = torch.tensor(charindices_pad).to(device)
    res = res.permute(1, 0, 2)
    
    return res
```

###(h)

在空文件highway.py 中，实现Highway网络，你可以继承nn.Module 类并把它命名为Highway。

- 您的模块将需要init() 和forward() 函数（其输入和输出您自己决定）
- forward() 函数将完成从$x_{conv\_out}$到$x_{highway}$映射的功能
- 注意，虽然上面的描述不是基于batch处理的，但你的forward()函数应该对数据进行批处理
- 确保你的模块使用了2个nn.Linear层

```python
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

#sigmoid函数

class Highway(nn.Module):
    def __init__(self, d):
        super(Highway, self).__init__()
        self.proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.a1 = nn.ReLU()
        self.a2 = nn.Sigmoid()
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x_proj = self.a1(self.proj(x))
        x_gate = self.a2(self.gate(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        
        return x_highway
```

上面的实现对应的公式如下：
$$
\begin{aligned}
\mathrm{x}_{\mathrm{proj}}&=\operatorname{Re} \mathrm{L} \mathrm{U}\left(\mathrm{W}_{\mathrm{proj}} \mathrm{x}_{\text {conv_out }}+\mathrm{b}_{\text {proj }}\right) \\
\mathrm{x}_{\text {gate }}&=\sigma\left(\mathrm{W}_{\text {gate }} \mathrm{X}_{\text {conv_out }}+\mathrm{b}_{\text {gate }}\right)\\
\mathrm{X}_{\mathrm{highway}}&=\mathrm{x}_{\mathrm{gate}} \odot \mathrm{x}_{\mathrm{proj}}+\left(1-\mathrm{x}_{\mathrm{gate}}\right) \odot \mathrm{x}_{\mathrm{conv}{\text{_out}} } 
\end{aligned}
$$

### (i)

在空的cnn.py文件中实现1个名为CNN的一维卷积类

```python
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, din, dout, k=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(din, dout, k)
    
    def forward(self, x):
        x_conv = self.conv(x)
        x_conv_temp = nn.ReLU()(x_conv)
        x_conv_out = nn.MaxPool1d(x_conv_temp.shape[-1])(x_conv_temp).squeeze(-1)
        
        return x_conv_out
```

### (j)

在model_embeddings.py中实现ModelEmbeddings类

```python
class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), 50, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.p = 0.3
        self.m_word = len(vocab.word2id)
        self.e_char = embed_size
        self.e_char = 50
        self.e_word = embed_size
        self.cnn = CNN(self.e_char, self.e_word)
        self.highway = Highway(self.e_word)
        self.dropout = nn.Dropout(self.p)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        
        x_emb = self.embeddings(input)
        sentence_length, batch_size, max_word_length, emb_size = x_emb.shape
        x_emb = x_emb.permute(0, 1, 3, 2)
        x_emb = x_emb.view(-1, emb_size, max_word_length)

        ### YOUR CODE HERE for part 1j
        x_conv = self.cnn(x_emb)
        x_high = self.highway(x_conv)
        x_word_emb = self.dropout(x_high)
        
        x_word_emb = x_word_emb.view(sentence_length, batch_size, -1)

        return x_word_emb
        ### END YOUR CODE
```

使用如下代码测试：

```
CODE
python sanity_check.py 1j
```

得到如下结果：

```
CODE
--------------------------------------------------------------------------------
Running Sanity Check for Question 1j: Model Embedding
--------------------------------------------------------------------------------
Sanity Check Passed for Question 1j: Model Embedding!
--------------------------------------------------------------------------------
```

### (k)

在nmt_model.py中，完成forward()函数，这次你需要进行字符级别的编码器实现。

```python
def forward(self, source: List[List[str] ], target: List[List[str] ]) -> torch.Tensor:
    """ Take a mini-batch of source and target sentences, compute the log-likelihood of
    target sentences under the language models learned by the NMT system.

    @param source (List[List[str] ]): list of source sentence tokens
    @param target (List[List[str] ]): list of target sentence tokens, wrapped by `<s>` and `</s>`

    @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                log-likelihood of generating the gold-standard target sentence for
                                each example in the input batch. Here b = batch size.
    """
    # Compute sentence lengths
    source_lengths = [len(s) for s in source]

    # Convert list of lists into tensors

    ## A4 code
    # source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
    # target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

    # enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
    # enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
    # combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
    ## End A4 code

    ### YOUR CODE HERE for part 1k
    ### TODO:
    ###     Modify the code lines above as needed to fetch the character-level tensor
    ###     to feed into encode() and decode(). You should:
    ###     - Keep `target_padded` from A4 code above for predictions
    ###     - Add `source_padded_chars` for character level padded encodings for source
    ###     - Add `target_padded_chars` for character level padded encodings for target
    ###     - Modify calls to encode() and decode() to use the character level encodings
    target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)
    source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)
    target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device=self.device)
    enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
    enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
    combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)
    ### END YOUR CODE

    P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

    # Zero out, probabilities for which we have nothing in the target text
    target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

    # Compute log probability of generating true target words
    target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
    scores = target_gold_words_log_prob.sum() # mhahn2 Small modification from A4 code.



    if self.charDecoder is not None:
        max_word_len = target_padded_chars.shape[-1]

        target_words = target_padded[1:].contiguous().view(-1)
        #print(max_word_len, target_padded_chars[1:].shape)
        #target_chars = target_padded_chars[1:].view(-1, max_word_len)
        target_chars = target_padded_chars[1:].reshape(-1, max_word_len)
        target_outputs = combined_outputs.view(-1, 256)

        target_chars_oov = target_chars #torch.index_select(target_chars, dim=0, index=oovIndices)
        rnn_states_oov = target_outputs #torch.index_select(target_outputs, dim=0, index=oovIndices)
        oovs_losses = self.charDecoder.train_forward(target_chars_oov.t(), (rnn_states_oov.unsqueeze(0), rnn_states_oov.unsqueeze(0)))
        scores = scores - oovs_losses

    return scores
```

在linux系统中可以采用下述的命令训练和测试

```shell
sh run.sh train_local_q1
sh run.sh test_local_q1
```

在windows下使用如下命令训练测试即可：

```
python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 --valid-niter=100 --max-epoch=101 --no-char-decoder

python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt --no-char-decoder
```

## part2 神经网络机器翻译 的 字符级别LSTM解码器

#### (a)

完成char_decoder.py文件中的CharDecoder.



根据公式：
$$
\begin{aligned}
\mathbf{h}_{t}, \mathbf{c}_{t}&=\text { CharDecoderLSTM }\left(\mathbf{x}_{t}, \mathbf{h}_{t-1}, \mathbf{c}_{t-1}\right)\\
\mathbf{s}_{t}&=\mathbf{W}_{\mathrm{dec}} \mathbf{h}_{t}+\mathbf{b}_{\mathrm{dec}}\\

\mathbf{p}_{t} &=\operatorname{softmax}\left(\mathbf{s}_{t}\right) \\
\text { loss char_dec } &=-\sum_{t=1}^{n} \log \mathbf{p}_{t}\left(x_{t+1}\right) 
\end{aligned}
$$
可以完成以下实现

```python
def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
    """ Init Character Decoder.

    @param hidden_size (int): Hidden size of the decoder LSTM
    @param char_embedding_size (int): dimensionality of character embeddings
    @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
    """
    ### YOUR CODE HERE for part 2a
    ### TODO - Initialize as an nn.Module.
    ###      - Initialize the following variables:
    ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
    ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
    ###        self.decoderCharEmb: Embedding matrix of character embeddings
    ###        self.target_vocab: vocabulary for the target language
    ###
    ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
    ###       - Set the padding_idx argument of the embedding matrix.
    ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
    super(CharDecoder, self).__init__()
    self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
    V_char = len(target_vocab.char2id)
    self.pad = target_vocab.char2id['<pad>']
    self.char_output_projection = nn.Linear(hidden_size, V_char)
    self.decoderCharEmb = nn.Embedding(V_char, char_embedding_size, self.pad)
    self.target_vocab = target_vocab
    ### END YOUR CODE
```

使用如下代码测试：

```shell
python sanity_check.py 2a
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 2a: CharDecoder.__init__()
--------------------------------------------------------------------------------
Sanity Check Passed for Question 2a: CharDecoder.__init__()!
--------------------------------------------------------------------------------
```

### (b)

完成char_decoder.py中的forward()函数

```python
def forward(self, input, dec_hidden=None):
    """ Forward pass of character decoder.

    @param input: tensor of integers, shape (length, batch)
    @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

    @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
    @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
    """
    ### YOUR CODE HERE for part 2b
    ### TODO - Implement the forward pass of the character decoder.
    X = self.decoderCharEmb(input)
    enc_hiddens, dec_hidden = self.charDecoder(X, dec_hidden)
    scores = self.char_output_projection(enc_hiddens)
    return scores, dec_hidden
    ### END YOUR CODE 
```

使用如下代码测试：

```
python sanity_check.py 2b
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 2b: CharDecoder.forward()
--------------------------------------------------------------------------------
Sanity Check Passed for Question 2b: CharDecoder.forward()!
--------------------------------------------------------------------------------
```

### (c)

完成char_decoder.py中的train_forward()函数

```python
def train_forward(self, char_sequence, dec_hidden=None):
    """ Forward computation during training.

    @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
    @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

    @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
    """
    ### YOUR CODE HERE for part 2c
    ### TODO - Implement training forward pass.
    ###
    ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
    ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
    scores, dec_hidden = self.forward(char_sequence, dec_hidden)
    scores = scores[:-1]
    target = char_sequence[1:]
    loss = 0
    batch_size = char_sequence.shape[1]
    loss_func = nn.CrossEntropyLoss(ignore_index=self.pad)
    for i in range(batch_size):
        loss += loss_func(scores[:, i], target[:, i])
    
    return loss
    ### END YOUR CODE
```

使用如下代码测试：

```
python sanity_check.py 2c
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 2c: CharDecoder.train_forward()
--------------------------------------------------------------------------------
Sanity Check Passed for Question 2c: CharDecoder.train_forward()!
--------------------------------------------------------------------------------
```

### (d)

完成char_decoder.py中的decode_greedy()函数

```python
def decode_greedy(self, initialStates, device, max_length=21):
    """ Greedy decoding
    @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
    @param device: torch.device (indicates whether the model is on CPU or GPU)
    @param max_length: maximum length of words to decode

    @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                          The decoded strings should NOT contain the start-of-word and end-of-word characters.
    """

    ### YOUR CODE HERE for part 2d
    ### TODO - Implement greedy decoding.
    ### Hints:
    ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
    ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
    ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
    ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
    batch_size = initialStates[0].shape[1]
    start = self.target_vocab.start_of_word
    end = self.target_vocab.end_of_word
    current_char = torch.tensor([ [start] * batch_size], device=device)
    dec_hidden = initialStates
    output = []
    char_index = []
    for i in range(max_length):
        scores, dec_hidden = self.forward(current_char, dec_hidden)
        current_char = torch.argmax(scores, axis=-1)
        char_index.append(current_char.detach().numpy().flatten())
    char_index = np.array(char_index)
    #截断
    for i in range(batch_size):
        index = char_index[:, i]
        tmp = ""
        for j in index:
            char = self.target_vocab.id2char[j]
            if j != end:
                tmp += char
            else:
                break
        output.append(tmp)
    
    return output
    
    ### END YOUR CODE
```

使用如下代码测试：

```
python sanity_check.py 2d
```

得到如下结果：

```
--------------------------------------------------------------------------------
Running Sanity Check for Question 2d: CharDecoder.decode_greedy()
--------------------------------------------------------------------------------
Sanity Check Passed for Question 2d: CharDecoder.decode_greedy()!
--------------------------------------------------------------------------------
```

### (e)

测试CharDecoder是否正确

在linux系统中可以采用下述的命令训练和测试

```shell
sh run.sh train_local_q2
sh run.sh test_local_q2
```

在windows下使用如下命令训练测试即可：

```
python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
```

### (f)

```
Corpus BLEU: 22.961492204691136
```

