# Assignment4讲解

Assignment4的任务相对前面的Assignment要更深入一些，是针对RNN和神经网络机器翻译的应用，可以视作一个相对完整的项目。

- [实验指导文档下载地址](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.pdf)
- [代码与数据下载地址](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.zip)

## RNN和神经网络机器翻译

机器翻译是指，构建一个系统完成源语言到目标语言的变换映射，比如给定一个源句子（比如西班牙语），输出一个目标句子（比如英语）。本次作业中要实现的是一个带注意力机制的Seq2Seq神经模型，用于构建神经网络机器翻译（NMT）系统。首先我们来看NMT系统的训练过程，它用到了双向LSTM作为编码器（encoder）和单向LSTM作为解码器（decoder）。

<img src="http://ww1.sinaimg.cn/large/0060yMmAly1gsjzdzvwbaj30pv0hzjuy.jpg" referrerpolicy="no-referrer"/>

给定长度为m的源语言句子（source），经过嵌入层，得到输入序列 $x_1, x_2, …, x_m \in R^{e \times 1}$，$e$是词向量大小。经过双向Encoder后，得到前向（→）和反向（←）LSTM的隐藏层和神经元状态，将两个方向的状态连接起来得到时间步 $i$ 的隐藏状态 $h_i^{enc}$ 和 $c_i^{enc}$ ：
$$
\mathbf{h}_{i}^{\text {enc }}=\left[\overleftarrow{\mathbf{h}_{i}^{\text {enc }}} ; \overrightarrow{\mathbf{h}_{i}^{\text {enc }}}\right] \text { where } \mathbf{h}_{i}^{\text {enc }} \in \mathbb{R}^{2 h \times 1}, \overleftarrow{\mathbf{h}_{i}^{\text {enc }}}, \overrightarrow{\mathbf{h}_{i}^{\text {en }}} \in \mathbb{R}^{h \times 1} \quad 1 \leq i \leq m
$$

$$
\mathbf{c}_{i}^{\text {enc }}=\left[\overleftarrow{\mathbf{c}_{i}^{\text {enc }}} ; \overrightarrow{\mathbf{c}_{i}^{\text {enc }}}\right] \text { where } \mathbf{c}_{i}^{\text {enc }} \in \mathbb{R}^{2 h \times 1}, \overleftarrow{\mathbf{c}_{i}^{\text {enc }}}, \overrightarrow{\mathbf{c}_{i}^{\text {en }}} \in \mathbb{R}^{h \times 1} \quad 1 \leq i \leq m
$$

接着我们使用一个线性层来初始化Decoder的初始隐藏、神经元的状态：
$$
\mathbf{h}_{0}^{\text {dec }}=\mathbf{W}_{h}\left[\overleftarrow{\mathbf{h}_{1}^{\text {enc }}} ; \overrightarrow{\mathbf{h}_{m}^{\text {enc }}}\right] \text { where } \mathbf{h}_{0}^{\text {dec }} \in \mathbb{R}^{h \times 1}, \mathbf{W}_{h} \in \mathbb{R}^{h \times 2 h}
$$

$$
\mathbf{c}_{0}^{\text {dec }}=\mathbf{W}_{c}\left[\overleftarrow{\mathbf{c}_{1}^{\text {enc }}} ; \overrightarrow{\mathbf{c}_{m}^{\text {enc }}}\right] \text { where } \mathbf{c}_{0}^{\text {dec }} \in \mathbb{R}^{h \times 1}, \mathbf{W}_{c} \in \mathbb{R}^{h \times 2 h}
$$



Decoder的时间步$t$ 的输入为 $\bar{y}_t$ ，它由目标语言句子 $y_t$和上一神经元的输出和上一神经元的输出$o_{t-1}$经过连接得到，经过连接得到，$o_0$是0向量，所以 $\bar{y}_t \in R^{(e + h) \times 1}$
$$
\mathbf{h}_{t}^{\mathrm{dec}}, \mathbf{c}_{t}^{\mathrm{dec}}=\operatorname{Decoder}\left(\overline{\mathbf{y}_{t}}, \mathbf{h}_{t-1}^{\mathrm{dec}}, \mathbf{c}_{t-1}^{\mathrm{dec}}\right) \text { where } \mathbf{h}_{t}^{\mathrm{dec}} \in \mathbb{R}^{h \times 1}, \mathbf{c}_{t}^{\mathrm{dec}} \in \mathbb{R}^{h \times 1}
$$
接着我们使用 $h^{dec}_t$ 来计算在 $h^{enc}_0, h^{enc}_1, …, h^{enc}_m$ 的乘积注意力（multiplicative attention）：
$$
\begin{array}{c}
\mathbf{e}_{t, i}=\left(\mathbf{h}_{t}^{\mathrm{dec}}\right)^{T} \mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }} \text { where } \mathbf{e}_{t} \in \mathbb{R}^{m \times 1}, \mathbf{W}_{\text {attProj }} \in \mathbb{R}^{h \times 2 h} \quad 1 \leq i \leq m \\
\alpha_{t}=\operatorname{softmax}\left(\mathbf{e}_{t}\right) \text { where } \alpha_{t} \in \mathbb{R}^{m \times 1} \\
\mathbf{a}_{t}=\sum_{i=1}^{m} \alpha_{t, i} \mathbf{h}_{i}^{\text {enc }} \text { where } \mathbf{a}_{t} \in \mathbb{R}^{2 h \times 1}
\end{array}
$$
然后将注意力 $\alpha_t$ 和解码器的隐藏状态 $h^{dec}_t$ 连接，送入线性层，得到 *combined-output* 向量 $o_t$
$$
\begin{array}{r}
\mathbf{u}_{t}=\left[\mathbf{a}_{t} ; \mathbf{h}_{t}^{\mathrm{dec}}\right] \text { where } \mathbf{u}_{t} \in \mathbb{R}^{3 h \times 1} \\
\mathbf{v}_{t}=\mathbf{W}_{u} \mathbf{u}_{t} \text { where } \mathbf{v}_{t} \in \mathbb{R}^{h \times 1}, \mathbf{W}_{u} \in \mathbb{R}^{h \times 3 h} \\
\mathbf{o}_{t}=\operatorname{dropout}\left(\tanh \left(\mathbf{v}_{t}\right)\right) \text { where } \mathbf{o}_{t} \in \mathbb{R}^{h \times 1}
\end{array}
$$
这样以来，目标词的概率分布则为：
$$
\mathbf{P}_{t}=\operatorname{softmax}\left(\mathbf{W}_{\text {vocab }} \mathbf{o}_{t}\right) \text { where } \mathbf{P}_{t} \in \mathbb{R}^{V_{t} \times 1}, \mathbf{W}_{\text {vocab }} \in \mathbb{R}^{V_{t} \times h}
$$
使用交叉熵做目标函数即可
$$
J_{t}(\theta)=\text { CrossEntropy }\left(\mathbf{P}_{t}, \mathbf{g}_{t}\right)
$$


代码实现部分，关键在于过程中的向量维度，向量维度匹配没有问题，整个过程的实现就比较清晰。

## part1 神经网络翻译系统代码实现

NMT的代码实现时间可能不久，但训练需要很长时间，这是因为神经网络翻译系统本身网络结构比较复杂，建议有条件的小伙伴在GPU上跑，如果没有自己的GPU，可以科学上网的同学也可以在[google colab](https://colab.research.google.com/)上试一下。

### (a)

```python
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str] ]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str] ]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_l = 0
    for sent in sents:
        max_l = max(max_l, len(sent))
    for sent in sents:
        new_sent = copy.deepcopy(sent)
        l = len(sent)
        while l < max_l:
            new_sent.append(pad_token)
            l += 1
        sents_padded.append(new_sent)

    ### END YOUR CODE

    return sents_padded
```

### (b)

model_embeddings.py的\__init__函数：

```python
def __init__(self, embed_size, vocab):
    """
    Init the Embedding layers.

    @param embed_size (int): Embedding size (dimensionality)
    @param vocab (Vocab): Vocabulary object containing src and tgt languages
                          See vocab.py for documentation.
    """
    super(ModelEmbeddings, self).__init__()
    self.embed_size = embed_size

    # default values
    self.source = None
    self.target = None

    src_pad_token_idx = vocab.src['<pad>']
    tgt_pad_token_idx = vocab.tgt['<pad>']

    ### YOUR CODE HERE (~2 Lines)
    ### TODO - Initialize the following variables:
    ###     self.source (Embedding Layer for source language)
    ###     self.target (Embedding Layer for target langauge)
    ###
    ### Note:
    ###     1. `vocab` object contains two vocabularies:
    ###            `vocab.src` for source
    ###            `vocab.tgt` for target
    ###     2. You can get the length of a specific vocabulary by running:
    ###             `len(vocab.<specific_vocabulary>)`
    ###     3. Remember to include the padding token for the specific vocabulary
    ###        when creating your Embedding.
    ###
    ### Use the following docs to properly initialize these variables:
    ###     Embedding Layer:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
    self.source = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
    self.target = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_token_idx)
```

### (c)

nmt_model.py的\__init__函数：

```python
def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
    """ Init NMT Model.

    @param embed_size (int): Embedding size (dimensionality)
    @param hidden_size (int): Hidden Size (dimensionality)
    @param vocab (Vocab): Vocabulary object containing src and tgt languages
                          See vocab.py for documentation.
    @param dropout_rate (float): Dropout probability, for attention
    """
    super(NMT, self).__init__()
    self.model_embeddings = ModelEmbeddings(embed_size, vocab)
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.vocab = vocab

    # default values
    self.encoder = None 
    self.decoder = None
    self.h_projection = None
    self.c_projection = None
    self.att_projection = None
    self.combined_output_projection = None
    self.target_vocab_projection = None
    self.dropout = None


    ### YOUR CODE HERE (~8 Lines)
    ### TODO - Initialize the following variables:
    ###     self.encoder (Bidirectional LSTM with bias)
    ###     self.decoder (LSTM Cell with bias)
    ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
    ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
    ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
    ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
    ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
    ###     self.dropout (Dropout Layer)
    ###
    ### Use the following docs to properly initialize these variables:
    ###     LSTM:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
    ###     LSTM Cell:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
    ###     Linear Layer:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    ###     Dropout Layer:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
    self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True)
    self.decoder = nn.LSTMCell((embed_size + hidden_size), hidden_size, bias=True)
    self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
    self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
    self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
    self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
    self.target_vocab_projection = nn.Linear(hidden_size, len(self.vocab.tgt))
    self.dropout = nn.Dropout(self.dropout_rate)

    ### END YOUR CODE
```

实现的过程中要特别关注维度的变化：

- 原始的数据是词索引，经过embedding，每个词变成了大小为 `embed_size` 的向量，所以encoder的输入大小为 `embed_size` ，隐藏层大小为 `hidden_size` 。

- decoder的输入是神经元输出和目标语言句子的嵌入向量，所以输入大小为 `embed_size + hidden_size`

  - $$
    \overline{\mathbf{y}_{t} } \in \mathbb{R}^{(e+h) \times 1}
    $$

- h_projection、c_projection的作用是将encoder的隐藏层状态降维，所以输入大小是 `2*hidden_size`，输出大小是`hidden_size`

- att_projection的作用也是降维，以便后续与decoder的隐藏层状态做矩阵乘法

- combined_output_projection的作用也将解码输出降维，输入是注意力向量和隐藏层状态连接得到的向量，大小为`3*hidden_size`，并保持输出大小为 `hidden_size`

- target_vocab_projection是将输出投影到词库的词中去

### (d)

nmt_model.py 的encode函数：

```python
def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] ]:
    """ Apply the encoder to source sentences to obtain encoder hidden states.
        Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

    @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                    b = batch_size, src_len = maximum source sentence length. Note that 
                                   these have already been sorted in order of longest to shortest sentence.
    @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
    @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                    b = batch size, src_len = maximum source sentence length, h = hidden size.
    @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                            hidden state and cell.
    """
    enc_hiddens, dec_init_state = None, None

    ### YOUR CODE HERE (~ 8 Lines)
    ### TODO:
    ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
    ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
    ###         that there is no initial hidden state or cell for the decoder.
    ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
    ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
    ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
    ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
    ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
    ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
    ###         - `init_decoder_hidden`:
    ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
    ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
    ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
    ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
    ###         - `init_decoder_cell`:
    ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
    ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
    ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
    ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
    ###
    ### See the following docs, as you may need to use some of the following functions in your implementation:
    ###     Pack the padded sequence X before passing to the encoder:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
    ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
    ###     Tensor Concatenation:
    ###         https://pytorch.org/docs/stable/torch.html#torch.cat
    ###     Tensor Permute:
    ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
    X = self.model_embeddings.source(source_padded)
    X_pack = nn.utils.rnn.pack_padded_sequence(X, source_lengths)
    enc_hiddens, (last_hidden, last_cell) = self.encoder(X_pack)
    enc_hiddens = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)[0]
    hdec = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
    cdec = self.c_projection(torch.cat((last_cell[0], last_cell[1]), 1))
    dec_init_state = (hdec, cdec)

    ### END YOUR CODE

    return enc_hiddens, dec_init_state
```

$$
\begin{aligned}
\mathbf{h}_{0}^{\mathrm{dec} } &=\mathbf{W}_{h}\left[\overleftarrow{\mathbf{h}_{1}^{\mathrm{enc} }} ; \overrightarrow{\mathbf{h}_{m}^{\mathrm{enc} }}\right] \\
\mathbf{c}_{0}^{\mathrm{dec} } &=\mathbf{W}_{c}\left[\overleftarrow{\mathrm{c}_{1}^{\mathrm{enc} }} ; \overrightarrow{ {\mathrm{c} }_{m}^{\mathrm{ent} }}\right]
\end{aligned}
$$

这里用到了 `pad` 和 `pack` 两个概念。

`pad` ：填充。将几个大小不一样的Tensor以最长的tensor长度为标准进行填充，一般是填充 `0`。

`pack`：打包。将几个 tensor打包成一个，返回一个`PackedSequence` 对象。

经过pack后，RNN可以对不同长度的样本数据进行小批量训练，否则就只能一个一个样本进行训练了。

`torch.cat` 可以将两个tensor拼接成一个



使用如下命令测试：

```
CODE
python sanity_check.py 1d
```

得到如下结果：

```
CODE
Running Sanity Check for Question 1d: Encode
--------------------------------------------------------------------------------
torch.Size([5, 20, 6]) torch.Size([5, 20, 6])
enc_hiddens Sanity Checks Passed!
dec_init_state[0] Sanity Checks Passed!
dec_init_state[1] Sanity Checks Passed!
--------------------------------------------------------------------------------
All Sanity Checks Passed for Question 1d: Encode!
--------------------------------------------------------------------------------
```

### (e)

nmt_model.py的decode函数：

```python
def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
            dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
    """Compute combined output vectors for a batch.

    @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                 b = batch size, src_len = maximum source sentence length, h = hidden size.
    @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                 b = batch size, src_len = maximum source sentence length.
    @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
    @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                   tgt_len = maximum target sentence length, b = batch size. 

    @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                    tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
    """
    # Chop of the <END> token for max length sentences.
    target_padded = target_padded[:-1]

    # Initialize the decoder state (hidden and cell)
    dec_state = dec_init_state

    # Initialize previous combined output vector o_{t-1} as zero
    batch_size = enc_hiddens.size(0)
    o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

    # Initialize a list we will use to collect the combined output o_t on each step
    combined_outputs = []

    ### YOUR CODE HERE (~9 Lines)
    ### TODO:
    ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
    ###         which should be shape (b, src_len, h),
    ###         where b = batch size, src_len = maximum source length, h = hidden size.
    ###         This is applying W_{attProj} to h^enc, as described in the PDF.
    ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
    ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
    ###     3. Use the torch.split function to iterate over the time dimension of Y.
    ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
    ###             - Squeeze Y_t into a tensor of dimension (b, e). 
    ###             - Construct Ybar_t by concatenating Y_t with o_prev.
    ###             - Use the step function to compute the the Decoder's next (cell, state) values
    ###               as well as the new combined output o_t.
    ###             - Append o_t to combined_outputs
    ###             - Update o_prev to the new o_t.
    ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
    ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
    ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
    ###
    ### Note:
    ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
    ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
    ###   
    ### Use the following docs to implement this functionality:
    ###     Zeros Tensor:
    ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
    ###     Tensor Splitting (iteration):
    ###         https://pytorch.org/docs/stable/torch.html#torch.split
    ###     Tensor Dimension Squeezing:
    ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
    ###     Tensor Concatenation:
    ###         https://pytorch.org/docs/stable/torch.html#torch.cat
    ###     Tensor Stacking:
    ###         https://pytorch.org/docs/stable/torch.html#torch.stack
    enc_hiddens_proj = self.att_projection(enc_hiddens) #(b, src_len, h)
    Y = self.model_embeddings.target(target_padded) #(tgt_len, b, e)
    Y_split = torch.split(Y, 1)
    for Y_t in Y_split:
        y_t = torch.squeeze(Y_t) #(b, e)
        Ybar_t = torch.cat((y_t, o_prev), dim=-1) #(b, e + h)
        dec_state, combined_output, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
        combined_outputs.append(combined_output)
        o_prev = combined_output
    combined_outputs = torch.stack(combined_outputs) #(tgt_len, b, h)

    ### END YOUR CODE

    return combined_outputs
```

$$
\mathbf{h}_{t}^{\mathrm{dec} }, \mathbf{c}_{t}^{\mathrm{dec} }=\operatorname{Decoder}\left(\overline{\mathbf{y}_{t} }, \mathbf{h}_{t-1}^{\mathrm{dec} }, \mathbf{c}_{t-1}^{\mathrm{dec} }\right)
$$

`torch.stack` 可以将一个list里的长度一样的tensor堆叠成一个tensor

`torch.squeeze` 可以将tensor里大小为1的维度给删掉，比如shape=(1,2,3) -> shape=(2,3)

使用如下命令测试：

```shell
CODE
python sanity_check.py 1d
```

得到如下结果：

```
CODE
--------------------------------------------------------------------------------
Running Sanity Check for Question 1e: Decode
--------------------------------------------------------------------------------
combined_outputs Sanity Checks Passed!
--------------------------------------------------------------------------------
All Sanity Checks Passed for Question 1e: Decode!
--------------------------------------------------------------------------------
```

### (f)

nmt_model.py的step函数：

```python
def step(self, Ybar_t: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
        enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
    """ Compute one forward step of the LSTM decoder, including the attention computation.

    @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                            where b = batch size, e = embedding size, h = hidden size.
    @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
            First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
    @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                src_len = maximum source length, h = hidden size.
    @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                where b = batch size, src_len = maximum source length, h = hidden size.
    @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                where b = batch size, src_len is maximum source length. 

    @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
            First tensor is decoder's new hidden state, second tensor is decoder's new cell.
    @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
    @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                            Note: You will not use this outside of this function.
                                  We are simply returning this value so that we can sanity check
                                  your implementation.
    """

    combined_output = None

    ### YOUR CODE HERE (~3 Lines)
    ### TODO:
    ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
    ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
    ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
    ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
    ###
    ###       Hints:
    ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
    ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
    ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
    ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
    ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
    ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
    ###
    ### Use the following docs to implement this functionality:
    ###     Batch Multiplication:
    ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
    ###     Tensor Unsqueeze:
    ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    ###     Tensor Squeeze:
    ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
    dec_state = self.decoder(Ybar_t, dec_state)
    dec_hidden, dec_cell = dec_state # (b, h)
    e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)) # (b, src_len, 1)
    e_t = torch.squeeze(e_t, dim=2) # (b, src_len)

    ### END YOUR CODE

    # Set e_t to -inf where enc_masks has 1
    if enc_masks is not None:
        #e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

    ### YOUR CODE HERE (~6 Lines)
    ### TODO:
    ###     1. Apply softmax to e_t to yield alpha_t
    ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
    ###         attention output vector, a_t.
    #$$     Hints:
    ###           - alpha_t is shape (b, src_len)
    ###           - enc_hiddens is shape (b, src_len, 2h)
    ###           - a_t should be shape (b, 2h)
    ###           - You will need to do some squeezing and unsqueezing.
    ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
    ###
    ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
    ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
    ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
    ###
    ### Use the following docs to implement this functionality:
    ###     Softmax:
    ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
    ###     Batch Multiplication:
    ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
    ###     Tensor View:
    ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
    ###     Tensor Concatenation:
    ###         https://pytorch.org/docs/stable/torch.html#torch.cat
    ###     Tanh:
    ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
    alpha_t = torch.unsqueeze(F.softmax(e_t, dim=1), dim=1)
    a_t = torch.squeeze(torch.bmm(alpha_t, enc_hiddens), dim=1) # (b, 2*h)
    u_t = torch.cat((a_t, dec_hidden), dim=1) # (b, 3*h)
    v_t = self.combined_output_projection(u_t) # (b, h)
    O_t = self.dropout(torch.tanh(v_t))

    ### END YOUR CODE

    combined_output = O_t
    return dec_state, combined_output, e_t
```

这里对应的公式为：

- 第1部分，注意力系数计算

$$
\begin{aligned}
\mathbf{e}_{t, i}&=\left(\mathbf{h}_{t}^{\mathrm{dec} }\right)^{T} \mathbf{W}_{\mathrm{att} } \mathrm{Proj} \mathbf{h}_{i}^{\mathrm{enc} } \\
\alpha_{t}&=\operatorname{Softmax}\left(\mathbf{e}_{t}\right) \\
\mathbf{a}_{t}&=\sum_{i}^{m} \alpha_{t, i} \mathbf{h}_{i}^{\mathrm{enc} }
\end{aligned}
$$

- 第2部分，计算注意力和combined output，并输出

$$
\begin{aligned}
\mathbf{u}_{t}&=\left[\mathbf{a}_{t} ; \mathbf{h}_{t}^{\mathrm{dec} }\right] \\
\mathrm{v}_{t}&=\mathbf{W}_{u} \mathbf{u}_{t} \\
\mathbf{o}_{t}&=\text { Dropout }\left(\tanh \left(\mathbf{v}_{t}\right)\right)\\
\mathbf{P}_{t}&=\operatorname{Softmax}\left(\mathbf{W}_{\mathrm{vocab} } \mathbf{o}_{t}\right)
\end{aligned}
$$

### (g)



```python
def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
    """ Generate sentence masks for encoder hidden states.

    @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
    
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, h = hidden size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, src_len:] = 1
    return enc_masks.to(self.device)
```

mask是为了记录哪些位置是pad，可以将单词和pad的α设置为很小的值，以此忽略pad的影响，将原始句子长度之后的位置都置为1即可。

### (h)

在命令行可以直接运行shell命令训练

```shell
sh run.sh train_local
```

windows上如果无法使用sh命令，所以使用如下方法训练：

```shell
python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
```

### (i)

使用如下方式测试：

```shell
sh run.sh test
```

windows上可以使用如下方法测试：

```shell
python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
```

### (j)

点乘注意力( dot product attention )计算快，但是可能不够准确，其他两种attention因为使用了权重参数，所以会更准确些。

## part2 NMT系统分析

### (a)

#### (i) 

- one of翻译成another，原因是语言限制，解决方法为增加类似的训练样本。

#### (ii) 

- 断句不对，原因是语言限制，解决方法为增加逗号较多的训练样本。

#### (iii) 

- 没有处理稀有词，原因是模型限制，应该增加处理稀有词的部分。

#### (iv) 

- an翻译成a，原因是模型限制，应该增加处理a,an的部分。

#### (v) 

- 老师翻译成teacher，原因是模型限制，应该减少模型对性别的偏差。

#### (vi) 

- 数量翻译错误，原因是模型限制，应该增加计量单位的转换。

### (b)

- 基于字符或者子词的embedding数量应该比单词级别的embedding小，因为它们更短，出现的频次更高，不同的单词可能是由同样一些字符和子词构成的。

### (c)

#### (i)

对于$c_1$：
$$
\begin{aligned}
p_1 &= \frac{0 + 1 + 1 + 1+0}{5}\\
&=\frac 3 5\\
p_2 &= \frac{0 + 1+ 1+0}{4}\\
&=\frac  1 2\\
r^\star &=5\\
c&=5\\
BLEU&=  \exp \left(\frac 1 2 \times \log \frac 3 5 + \frac 1 2 \times \log \frac 1 2 \right)\\
&=0.5477225575051662
\end{aligned}
$$
对于$c_2$：
$$
\begin{aligned}
p_1 &= \frac{1 + 1 + 0 + 1+1}{5}\\
&=\frac 4 5\\
p_2 &= \frac{1+ 0+ 0+1}{4}\\
&=\frac  1 2\\
r^\star &= 4\\
c&=5 \\
BLEU&= \exp \left(\frac 1 2 \times \log \frac 4 5 + \frac 1 2 \times \log \frac 1 2 \right)\\
&=0.6324555320336759
\end{aligned}
$$
根据BLEU，$c_2$比$c_1$好，对比原句，我觉得这是合理的。

#### (ii)

对于$c_1$：
$$
\begin{aligned}p_1 &= \frac{0 + 1 + 1 + 1+0}{5}\\
&=\frac 3 5\\p_2 &= \frac{0 + 1+ 1+0}{4}\\
&=\frac  1 2\\
r^\star &= 5\\
c&=5\\
BLEU&=  \exp \left(\frac 1 2 \times \log \frac 3 5 + \frac 1 2 \times \log \frac 1 2 \right)\\&=0.5477225575051662\end{aligned}
$$
对于$c_2$：
$$
\begin{aligned}
p_1 &= \frac{1 + 1 + 0 + 0+0}{5}\\
&=\frac2 5\\
p_2 &= \frac{1+ 0+ 0+0}{4}\\
&=\frac  1 4\\
r^\star &= 5\\
c&=5 \\
BLEU&= \exp \left(\frac 1 2 \times \log \frac 2 5 + \frac 1 2 \times \log \frac 1 4 \right)\\
&=0.316227766016838
\end{aligned}
$$
根据BLEU，$c_1$比$c_2$好，对比原句，我觉得这是不合理的。

#### (iii)

一个翻译可能不够准确，参考多个翻译的结果更加好。

#### (iv)

优势：可以较为准确的评估翻译结果，定量化。

劣势：计算速度慢，需要人为提供多个翻译。