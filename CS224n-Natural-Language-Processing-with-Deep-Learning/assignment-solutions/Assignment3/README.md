# Assignment3讲解

[Assignment3](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a3.zip)(点击下载) 的任务包括对通用神经网络技巧（Adam优化器和Dropout处理）的理解，以及代码完成神经网络依存解析器(Neural Transition-Based Dependency Parsing)。


## part1 机器学习和神经网络
关于Adam和Dropout的理解，除了直接根据[Assignment3文档](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a3.pdf)理解，还可以参考原始论文。

- [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

参考解答：

### (a)

(i) 

该更新方式实际上计算了梯度的加权和，所以不会变化太大；低方差可以减少震荡的情形。

(ii) 

梯度小的地方会得到更大的更新，梯度大的地方会得到更小的更新，该方法使得各个方向的更新幅度比较接近，可以减少震荡的情形。

### (b)

(i) 
$$
\begin{aligned}
\mathbb{E}_{p_{\text {drop }}}\left[\mathbf{h}_{\text {drop}}\right]_{i}
&=\gamma (1- p_{\text{drop}}) h_i \\ 
&=h_{i}

\end{aligned}
$$
所以有
$$
\gamma = \frac 1 {1- p_{\text{drop}}}
$$


(ii)

训练时使用dropout是为了通过随机失活处理探索更多网络结构，增加模型的泛化性；评估时每一个神经元的信息都是有用的，无法随机丢弃，所以不使用dropout，以得到稳定准确的结果。



## part2 基于神经Transition的依存句法分析

### (a)

依存句法分析，就是分析句子的句法结构，建立 *head* 词和修饰这些head的词之间的关系。这次构建的是 *transition-based* 解析器，它增量的，每一次只进行一步解析动作来生成依存关系，每一步解析称为 *partial parse*，可表示为：

- 一个 *stack* ，已被处理的词
- 一个 *buffer* ，待处理的词
- 一个 *dependencies* ，解析器生成的依赖

初始状态下，stack里有只 ROOT 一个词，在每一次解析中，运行 *transition* 操作，分为三个类型：

- SHIFT：将buffer的左边（头部）第一个词取出，放到stack的右边（尾部）
- LEFT-ARC：将stack的右第二个词作为依赖项，它依赖于右边第一个词，生成一个依赖关系，并删除右第二个词。
- RIGHT-ARC：将stack的右第一个词作为依赖项，它依赖于右边第二个词，生成一个依赖关系，并删除右第一个词。

当buffer长度为0，stack长度为1（只有ROOT）时就算解析完毕了。

| Stack                        | Buffer                                 | New dependency   | Transition            |
| ---------------------------- | -------------------------------------- | ---------------- | --------------------- |
| [ROOT]                       | [I, parsed, this, sentence, correctly] |                  | Initial Configuration |
| [ROOT, I]                    | [parsed, this, sentence, correctly]    |                  | SHIFT                 |
| [ROOT, I, parsed]            | [this, sentence, correctly]            |                  | SHIFT                 |
| [ROOT, parsed]               | [this, sentence, correctly]            | parsed→I         | LEFT-ARC              |
| [ROOT, parsed,this]          | [sentence, correctly]                  |                  | SHIFT                 |
| [ROOT, parsed,this,sentence] | [correctly]                            |                  | SHIFT                 |
| [ROOT, parsed,sentence]      | [correctly]                            | sentence→this    | LEFT-ARC              |
| [ROOT, parsed]               | [correctly]                            | parsed→sentence  | RIGHT-ARC             |
| [ROOT, parsed,correctly]     | []                                     |                  | SHIFT                 |
| [ROOT, parsed]               | []                                     | parsed→correctly | RIGHT-ARC             |
| [ROOT]                       | []                                     | ROOT→parsed      | RIGHT-ARC             |

### (b) **长度为n的句子，经过多少步后可以被解析完（用n表示）？简要解析为什么**

答：要使buffer长度为0，则需要n步，使stack长度为1，也需要n步，所以经过2n步后解析完毕。

### (c) 完成parser_trainsitions.py

#### init

初始化函数

```python
### YOUR CODE HERE (3 Lines)
### Your code should initialize the following fields:
###     self.stack: The current stack represented as a list with the top of the stack as the
###                 last element of the list.
###     self.buffer: The current buffer represented as a list with the first item on the
###                  buffer as the first item of the list
###     self.dependencies: The list of dependencies produced so far. Represented as a list of
###             tuples where each tuple is of the form (head, dependent).
###             Order for this list doesn't matter.
###
### Note: The root token should be represented with the string "ROOT"
###
self.stack = ["ROOT"]
self.buffer = copy.deepcopy(sentence)
self.dependencies = []


### END YOUR CODE
```

#### parse_step

注意，stack的栈顶是list的右边，buffer队头是list的左边

```python

### YOUR CODE HERE (~7-10 Lines)
### TODO:
###     Implement a single parsing step, i.e. the logic for the following as
###     described in the pdf handout:
###         1. Shift
###         2. Left Arc
###         3. Right Arc
if transition == "S":
word = self.buffer.pop(0)
self.stack.append(word)
elif transition == "LA":
self.dependencies.append((self.stack[-1], self.stack[-2]))
self.stack.pop(-2)
else:
self.dependencies.append((self.stack[-2], self.stack[-1]))
self.stack.pop(-1)

### END YOUR CODE
```

#### minibatch_parse

sentences含多个句子，每个句子都有一个partial parse对象。所以每一次取出一个batch的parse来进行一次transition操作，同时要过滤掉已经完成的parse。

```python

### YOUR CODE HERE (~8-10 Lines)
### TODO:
###     Implement the minibatch parse algorithm as described in the pdf handout
###
###     Note: A shallow copy (as denoted in the PDF) can be made with the "=" sign in python, e.g.
###                 unfinished_parses = partial_parses[:].
###             Here `unfinished_parses` is a shallow copy of `partial_parses`.
###             In Python, a shallow copied list like `unfinished_parses` does not contain new instances
###             of the object stored in `partial_parses`. Rather both lists refer to the same objects.
###             In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
###             contains references to the same objects. Thus, you should NOT use the `del` operator
###             to remove objects from the `unfinished_parses` list. This will free the underlying memory that
###             is being accessed by `partial_parses` and may cause your code to crash.
partial_parses = [PartialParse(sentence) for sentence in sentences]
unfinished_parses = partial_parses[:]
n = len(unfinished_parses)

while (n > 0):
    l = min(n, batch_size)
    transitions = model.predict(unfinished_parses[:l])
    for parse, trans in zip(unfinished_parses[:l], transitions):
        parse.parse_step(trans)
        if len(parse.stack) == 1:
            unfinished_parses.remove(parse)
            n -= 1
dependencies = [partial_parses.dependencies for partial_parses in partial_parses]

### END YOUR CODE
```



### (e) 完成 parser_model.py

实质上就是搭建一个三层的前馈神经网络，用ReLU做激活函数，最后一层用softmax输出，交叉熵做损失函数，同时还加了embedding层

#### init

初始化三个层，`n_features` 表示每一个词用几个特征来表示，每一个特征都要embed，所以输入层的大小是 `n_features * embed_size` 。

```python
# Input Layer
self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size)
nn.init.xavier_uniform_(self.embed_to_hidden.weight)

# Dropout Layer
self.dropout = nn.Dropout(self.dropout_prob)

# Output Layer
self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
nn.init.xavier_uniform_(self.hidden_to_logits.weight)
```

#### embedding_lookup

使用的是预训练的embedding(Collobert at. 2011)

```python
x = self.pretrained_embeddings(t)
x = x.view(x.shape[0], -1)
```

#### forward

提取特征、输入网络拿到节点，这里没用加softmax层是因为 [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) 会内部帮我们加

```python
embeddings = self.embedding_lookup(t)
hidden = self.embed_to_hidden(embeddings)
hidden = nn.ReLU()(hidden)
hidden = self.dropout(hidden)
logits = self.hidden_to_logits(hidden)
```

#### train

```python
optimizer = optim.Adam(parser.model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
```

#### train_for_epoch

```python
logits = parser.model(train_x)
loss = loss_func(logits, train_y)

loss.backward()
optimizer.step()
```