# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import tensorflow as tf

if sys.version_info[0] > 2:
    is_py3 = True
else:
    # reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(content)
                if content:

                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

def read_file_nolabel(filename):
    """读取文件数据"""
    contents = []
    with open_file(filename) as f:
        for line in f:
            try:
                content = line.strip()
                contents.append(content)
                # if content:
                #     contents.append(list(native_content(content)))
                    # labels.append(native_content(label))
            except:
                pass
    return contents

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储, x， y"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    # with open('./data/word.txt', 'w') as out:
    #     for i in range(len(all_data)):
    #         out.write(all_data[i] + ' ')
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def load_word2vec_embedding(word_embedding_file, vocab_size, embedding_dim):
    '''
        加载外接的词向量。
        :return:
    '''
    print ('loading word embedding, it will take few minutes...')
    embeddings = np.random.uniform(-1,1,(vocab_size, embedding_dim))  # 4223, 300
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(embedding_dim)))    # 300
    # padding = np.asarray(rng.normal(size=(embedding_dim)))

    f = open(word_embedding_file)
    for index, line in enumerate(f):
        values = line.split()
        try:
            coefs = np.asarray(values[1:], dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print (values[0], values[1:])

        embeddings[index] = coefs   # 将词和对应的向量存到字典里
    f.close()


    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    # embeddings[-2] = unknown
    # embeddings[-1] = unknown

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size, embedding_dim],
                           initializer=tf.constant_initializer(embeddings), trainable=False)

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['价格', '动力', '油耗', '操控', '配置', '舒适性', '安全性', '内饰', '外观', '空间']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


# def process_file(filename, word_to_id, cat_to_id, max_length=600):
def process_file(filename, cat_to_id):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    # data_id, label_id = [], []
    label_id = []
    for i in range(len(contents)):
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return contents, y_pad

def process_file_nolabel(filename, word_to_id, max_length=600):
    """将文件转换为id表示"""
    contents = read_file_nolabel(filename)

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    # return x_pad
    return contents


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    # 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    # 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id:end_id], y[start_id:end_id]

def attention(inputs, attention_size, l2_reg_lambda):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        inputs = tf.concat(2, inputs)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    #if l2_reg_lambda > 0:
    #    l2_loss += tf.nn.l2_loss(W_omega)
    #    l2_loss += tf.nn.l2_loss(b_omega)
    #    l2_loss += tf.nn.l2_loss(u_omega)
    #    tf.add_to_collection('losses', l2_loss)

    return output