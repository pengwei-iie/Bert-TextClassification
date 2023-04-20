# Text Classification with RNN--2018CCFBDCI汽车用户观点提取

汽车用户观点提取，使用bert模型的词向量作为RNN的初始化，其中data的train_x.npy表示的是bert的输入格式
而原始的数据集是经过word2id以及padding的，y不需要变化，rnn和加bert的rnn都可以用。具体参考text_Loader下的process file函数。


使用循环神经网络进行中文文本分类

## 环境

- Python 2/3 
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy

## 数据集

使用汽车用户观点提取的任务进行训练与测试，数据集请自行到2018CCFBCI(https://www.datafountain.cn/competitions/329/details)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类

## 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

## RNN循环神经网络

### 配置项

RNN可配置的参数如下所示，在`rnn_model.py`中。

```python
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
```

### RNN-bert模型

具体参看`run_rnn_bert.py`的实现。

关于RNN-bert模型--清华新浪新闻数据集的实现见github（https://github.com/a414351664/Bert-THUCNews ）具体如下！

# Text Classification with RNN
有bert后缀的都是在原来的基础上，进行改造，得到的效果能到97.5
无bert后缀的加入了Attention以及多层或者双向模型

使用循环神经网络进行中文文本分类


## 环境

- Python 2/3
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类5000条数据。相关数据下载地址：链接: https://pan.baidu.com/s/1GmBFZfDKsXBMFEYQWFdrfQ 提取码: 9ixg 

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

数据集划分如下：

- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

从原数据集生成子集的过程请参看`helper`下的两个脚本。其中，`copy_data.sh`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行该文件后，得到三个数据文件：

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## RNN循环神经网络

### 配置项

RNN可配置的参数如下所示，在`rnn_model.py`中。

```python
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 768  # 词向量维度
    seq_length = 512  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表达小

    num_layers = 1  # 隐藏层层数
    hidden_dim = 512  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    attention_dim = 512
    l2_reg_lambda = 0.01

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
```

### RNN模型

具体参看`rnn_model_bert.py`的实现。

大致结构如下：

![images/rnn_architecture](images/rnn_architecture.png)



