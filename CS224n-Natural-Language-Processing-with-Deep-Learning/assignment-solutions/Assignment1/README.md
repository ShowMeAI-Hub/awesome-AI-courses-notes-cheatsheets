# Assignment1讲解

[Assignment1](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a1.zip)(点击下载) 和前两讲的lecture内容是对应的，任务是探索词向量。内容包括 两种方式实现词向量：**基于计数的共现矩阵** 和 **基于预测的word2vec**、对应的词向量相似度计算、研究近义词、反义词等。

作业是ipynb格式文件，可以用jupyter打开，本地打开可以安装anaconda启动jupyter，可以科学上网的小伙伴也可以使用[google colab](https://colab.research.google.com/)上传Notebook运行。

## 词向量

词向量是对文本的向量化表征，也可以直接用于下游NLP任务(如问答、文本生成、翻译等) ，词向量包含丰富的词汇含义分布信息，它的好坏能在很大程度上影响下游任务的性能。本次作业探索两类词向量：*共现矩阵* 和 *word2vec* 。

**术语解释：** “word vectors” 和 “word embeddings” 在很多场景下是一个意思，都表示词向量。”embedding” 这个词的内在含义是将词编码到一个底维空间中。”概念上而言，它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。”——[维基百科](https://zh.wikipedia.org/wiki/词嵌入)

## Part 1：基于计数的词向量

大多数词向量模型都是基于一个观点：

**You shall know a word by the company it keeps ([Firth, J. R. 1957:11](https://en.wikipedia.org/wiki/John_Rupert_Firth))**

大多数词向量的实现的核心是 *相似词* ，也就是同义词，因为它们有相似的上下文。这里我们介绍一种策略叫做 *共现矩阵* (更多信息可以查看 [这里](http://web.stanford.edu/class/cs124/lec/vectorsemantics.video.pdf) 或 [这里](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285) )

这部分要实现的是，给定语料库，根据共现矩阵计算词向量，得到语料库中每个词的词向量，流程如下：

- 计算语料库的单词集
- 计算共现矩阵
- 使用SVD降维
- 分析词向量

### 问题1.1：实现 dicintct_words

计算语料库的单词数量、单词集

```python
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    corpus_words =  sorted(list(set([word for sentence in corpus for word in sentence])))
    num_corpus_words = len(corpus_words)

    # ------------------

    return corpus_words, num_corpus_words
```

### 问题1.2：实现compute_co_occurrence_matrix

计算给定语料库的共现矩阵。具体来说，对于每一个词 `w`，统计前、后方 `window_size` 个词的出现次数

```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = dict([(word, index) for index, word in enumerate(words)])
    
    # ------------------
    # Write your implementation here.
    for sentence in corpus:
        current_index = 0
        sentence_len = len(sentence)
        indices = [word2Ind[i] for i in sentence]
        while current_index < sentence_len:
            left  = max(current_index - window_size, 0)
            right = min(current_index + window_size + 1, sentence_len) 
            current_word = sentence[current_index]
            current_word_index = word2Ind[current_word]
            words_around = indices[left:current_index] + indices[current_index+1:right]
            
            for ind in words_around:
                M[current_word_index, ind] += 1
            
            current_index += 1

    # ------------------

    return M, word2Ind
```

### 问题1.3：实现 reduce_to_k_dim

这一步是降维。在问题1.2得到的是一个N x N的矩阵（N是单词集的大小），使用scikit-learn实现的SVD（奇异值分解），从这个大矩阵里分解出一个含k个特制的N x k 小矩阵。

**注意**：在numpy、scipy和scikit-learn都提供了一些SVD的实现，但是只有scipy、sklearn有Truncated SVD，并且只有sklearn提供了计算大规模SVD的高效的randomized算法，详情参考[sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) 。

```python
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)

    # ------------------

    print("Done.")
    return M_reduced
```

### 问题1.4 实现 plot_embeddings

基于matplotlib，用`scatter` 画 “×”，用 `text` 写字

```python
def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    
    for w in words:
        index = word2Ind[w]
        embedding = M_reduced[index]
        x, y  = embedding[0], embedding[1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.show()
    # ------------------
```

效果：

<img src="http://ww1.sinaimg.cn/large/0060yMmAly1gsz23kkkuej30gv08i74g.jpg" referrerpolicy="no-referrer" />

### 问题1.5：共现打印分析

将词嵌入到2个维度上，归一化，最终词向量会落到一个单位圆内，在坐标系上寻找相近的词。

<img src="http://ww1.sinaimg.cn/large/0060yMmAly1gsz2ozcoccj30gy08imxe.jpg" referrerpolicy="no-referrer" />

## Part 2：基于预测的词向量

目前，基于预测的词向量是最流行的，比如word2vec。现在我们来探索word2vec生成的词向量，如果想要深入了解，可以读一读 [原始论文](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 。

这一部分主要是使用gensim探索词向量，不是自己实现word2vec，所使用的词向量维度是300，由google发布。

首先使用SVD降维，将300维降2维，方便打印查看。

### 问题2.1：word2vec打印分析

和问题1.5一样

### 问题2.2：一词多义

找到一个有多个含义的词（比如 “leaves”，”scoop”），这种词的top-10相似词（根据余弦相似度）里有两个词的意思不一样。比如”leaves”（叶子，花瓣）的top-10词里有”vanishes”（消失）和”stalks”（茎秆）。

这里我找到的词是”column”（列），它的top-10里有”columnist”（专栏作家）和”article”（文章）

```python
# ------------------
# Write your polysemous word exploration code here.

wv_from_bin.most_similar("column")

# ------------------
```

输出：

```
[('columns', 0.767943263053894),
 ('columnist', 0.6541407108306885),
 ('article', 0.651928186416626),
 ('columnists', 0.617466926574707),
 ('syndicated_column', 0.599014401435852),
 ('op_ed', 0.588202714920044),
 ('Op_Ed', 0.5801560282707214),
 ('op_ed_column', 0.5779396891593933),
 ('nationally_syndicated_column', 0.572504997253418),
 ('colum', 0.5595961213111877)]
```

### 问题2.3：近义词和反义词

找到三个词(w1, w2, w3)，其中w1和w2是近义词，w1和w3是反义词，但是w1和w3的距离\<w1和w2的距离。例如：w1=”happy”，w2=”cheerful”，w3=”sad”

为什么反义词的相似度反而更大呢（距离越小说明越相似）？因为他们的上下文通常非常一致

```python
# ------------------
# Write your synonym & antonym exploration code here.

w1 = "love"
w2 = "like"
w3 = "hate"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# ------------------
```

输出：

```
Synonyms love, like have cosine distance: 0.6328612565994263
Antonyms love, hate have cosine distance: 0.39960432052612305
```

### 问题2.4：类比

man 对于 king，相当于woman对于___，这样的问题也可以用word2vec来解决，关于most_similar的详细用法可以参考 [GenSim文档](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar)。

这里我们找另外一组类比

```python
# ------------------
# Write your analogy exploration code here.
# man : him :: woman : her
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'him'], negative=['man']))

# ------------------
```

输出：

```
[('her', 0.694490909576416),
 ('she', 0.6385233402252197),
 ('me', 0.628451406955719),
 ('herself', 0.6239798665046692),
 ('them', 0.5843966007232666),
 ('She', 0.5237804651260376),
 ('myself', 0.4885627031326294),
 ('saidshe', 0.48337966203689575),
 ('he', 0.48184287548065186),
 ('Gail_Quets', 0.4784894585609436)]
```

可以看到正确的计算出了”queen”

### 问题2.5：错误的类比

找到一个错误的类比，树：树叶 ：：花：花瓣

```python
# ------------------
# Write your incorrect analogy exploration code here.
# tree : leaf :: flower : petal
pprint.pprint(wv_from_bin.most_similar(positive=['leaf', 'flower'], negative=['tree']))

# ------------------
```

输出：

```
[('floral', 0.5532568693161011),
 ('marigold', 0.5291938185691833),
 ('tulip', 0.521312952041626),
 ('rooted_cuttings', 0.5189826488494873),
 ('variegation', 0.5136324763298035),
 ('Asiatic_lilies', 0.5132641792297363),
 ('gerberas', 0.5106234550476074),
 ('gerbera_daisies', 0.5101010203361511),
 ('Verbena_bonariensis', 0.5070016980171204),
 ('violet', 0.5058108568191528)]
```

结果输出的里面没有“花瓣”

### 问题2.6：偏见分析

注意偏见是很重要的比如性别歧视、种族歧视等，执行下面代码，分析两个问题：

(a) 哪个词与“woman”和“boss”最相似，和“man”最不相似?

(b) 哪个词与“man”和“boss”最相似，和“woman”最不相似?

```python
# Run this cell
# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be
# most dissimilar from.
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'boss'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'boss'], negative=['woman']))
```

输出：

```
[('bosses', 0.5522644519805908),
 ('manageress', 0.49151360988616943),
 ('exec', 0.45940813422203064),
 ('Manageress', 0.45598435401916504),
 ('receptionist', 0.4474116563796997),
 ('Jane_Danson', 0.44480544328689575),
 ('Fiz_Jennie_McAlpine', 0.44275766611099243),
 ('Coronation_Street_actress', 0.44275566935539246),
 ('supremo', 0.4409853219985962),
 ('coworker', 0.43986251950263977)]

[('supremo', 0.6097398400306702),
 ('MOTHERWELL_boss', 0.5489562153816223),
 ('CARETAKER_boss', 0.5375303626060486),
 ('Bully_Wee_boss', 0.5333974361419678),
 ('YEOVIL_Town_boss', 0.5321705341339111),
 ('head_honcho', 0.5281980037689209),
 ('manager_Stan_Ternent', 0.525971531867981),
 ('Viv_Busby', 0.5256162881851196),
 ('striker_Gabby_Agbonlahor', 0.5250812768936157),
 ('BARNSLEY_boss', 0.5238943099975586)]
```

第一个类比 男人:女人 :: 老板:___，最合适的词应该是”landlady”（老板娘）之类的，但是top-10里只有”manageress”（女经理），”receptionist”（接待员）之类的词。

第二个类比 女人:男人 :: 老板:___，输出的不知道是些什么东西/捂脸

### 问题2.7：自行分析偏见

这里我找的例子是：

- 男人:女人 :: 医生:___
- 女人:男人 :: 医生:___

```python
# ------------------
# Write your bias exploration code here.

pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'doctor'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'doctor'], negative=['woman']))

# ------------------
```

输出：

```
[('gynecologist', 0.7093892097473145),
 ('nurse', 0.647728681564331),
 ('doctors', 0.6471461057662964),
 ('physician', 0.64389967918396),
 ('pediatrician', 0.6249487996101379),
 ('nurse_practitioner', 0.6218312978744507),
 ('obstetrician', 0.6072014570236206),
 ('ob_gyn', 0.5986712574958801),
 ('midwife', 0.5927063226699829),
 ('dermatologist', 0.5739566683769226)]

[('physician', 0.6463665962219238),
 ('doctors', 0.5858404040336609),
 ('surgeon', 0.5723941326141357),
 ('dentist', 0.552364706993103),
 ('cardiologist', 0.5413815975189209),
 ('neurologist', 0.5271126627922058),
 ('neurosurgeon', 0.5249835848808289),
 ('urologist', 0.5247740149497986),
 ('Doctor', 0.5240625143051147),
 ('internist', 0.5183224081993103)]
```

第一个类比中，我们看到了”nurse”（护士），这是一个有偏见的类比

### 问题2.8：思考偏见问题

什么会导致词向量里的偏见？

因为数据集中有偏见

## 参考

[1] CS224n: Natural Language Processing with Deep Learning, 2019-03-12. http://web.stanford.edu/class/cs224n.