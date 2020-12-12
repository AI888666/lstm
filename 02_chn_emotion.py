# 中文文本情感分析(利用LSTM模型实现)
# 数据集：关于酒店的中文评论
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import random
from multiprocessing import cpu_count

# 公共变量
mydict = {}  # 存放文字及编码
code = 1
data_file = "data/hotel_discuss2.csv"  # 原始样本文件路径
dict_file = "data/horel_dict.txt"  # 字典文件路径
encoding_file = "data/horel_encoding.txt"  # 编码后的样本文件路径
puncts = "\n"  # 需要过滤掉的标点符号列表

# 预处理
with open(data_file, "r", encoding="utf-8-sig") as f:
    for line in f.readlines():
        trim_line = line.strip()

        for ch in trim_line:  # 符号不参与编码
            if ch in puncts:
                continue
            if ch in mydict:  # 字符已经在字典中
                continue
            else:
                mydict[ch] = code  # 分配一个编码， 并将键值对存入字典中
                code += 1
    code += 1
    mydict["<unk>"] = code  # 未知字符编码

# 将字典存入文件中
with open(dict_file, "w", encoding="utf-8-sig") as f:
    f.write(str(mydict))
    print("数据字典保存成功。")


# 加载字典文件中的内容
def load_dict():
    with open(dict_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return new_dict


# 对样本中的评论部分进行编码
new_dict = load_dict()

with open(data_file, "r", encoding="utf-8-sig") as f:
    with open(encoding_file, "w", encoding="utf-8-sig") as fw:
        for line in f.readlines():  # 遍历原始样本每一行
            label = line[0]  # 标签
            remark = line[1:-1]  # 评论

            for ch in remark:  # 遍历评论，对每一个文字进行编码
                if ch in puncts:  # 过滤掉不参与编码的符号
                    continue
                else:
                    fw.write(str(mydict[ch]))  # 写入一个编码值
                    fw.write(",")  # 再写入一个逗号
            fw.write("\t" + str(label) + "\n")  # 在评论编码字符串后追加tab分隔符、类别、换行符
print("数据预处理完成。")


###########################################
# 获取字典长度
def get_dict_len(dict_path):
    with open(dict_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return len(new_dict.keys())


# 创建reader函数
def data_mapper(sample):
    dt, lbl = sample
    # 取出每个编码值，并转换为整型，构造列表返回
    val = [int(word) for word in dt.split(",") if word.isdigit()]
    return val, int(lbl)  # 返回整型列表、类别


def train_reader(train_list_path):
    def reader():
        with open(train_list_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱样本数据

            for line in lines:
                data, lbael = line.split("\t")
                yield data, label

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# 定义LSTM网络模型
def lstm_net(ipt, input_dim):
    ipt = fluid.layers.reshape(ipt, [-1, 1],
                               inplace=True)  # 是否替换，True则表示输入，返回是同一个独享
    # 词嵌入层
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128],
                                 is_sparse=True)  # 是否表示为稀疏数据格式
    # 全链接层
    fc1 = fluid.layers.fc(input=emb, size=128)

    # 第一分支：LSTM
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, size=128)
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type="max")

    # 第二分支：sequence pool
    spool = fluid.layers.sequence_pool(input=fc1, pool_type="max")

    # 输出层
    out = fluid.layers.fc(input=[spool, lstm2],  # 前面两个分支输出作为输入
                          size=2,  # 输出类别2个：正面、反面
                          act="softmax")
    return out


# 定义数据
dict_len = get_dict_len(dict_file)  # 获取字典长度
# 变量
rmk = fluid.layers.data(name="rmk", shape=[1], dtype="int64", lod_level=1)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# 创建模型
model = lstm_net(rmk, dict_len)

# 损失函数
cost = fluid.layers.cross_entropy(input=model,  # 预测值
                                  label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)  # 对损失函数求均值作为优化的目标函数
# 准确率
acc = fluid.layers.accuracy(input=model, label=label)
# 优化器
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化
# reader
reader = train_reader(encoding_file)
batch_train_reader = paddle.batch(reader, batch_size=128)
# feeder
feeder = fluid.DataFeeder(place=place, feed_list=[rmk, label])

# 迭代训练
for pass_id in range(5):
    for batch_id, data in enumerate(batch_train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        if batch_id % 20 == 0:
            print("pass_id：%d, batch_id：%d, cost：%f, acc：%f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

print("模型训练完成。")

# 保存模型
model_save_dir = "model/chn_emotion/"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(model_save_dir,
                              feeded_var_names=[rmk.name],
                              target_vars=[model],
                              executor=exe)
print("模型保存完成，保存路径：", model_save_dir)

import paddle
import paddle.fluid as fluid
import numpy as np
import os
import random
from multiprocessing import cpu_count

data_file = "data/hotel_discuss2.csv"  # 原始样本文件路径
dict_file = "data/horel_dict.txt"  # 字典文件路径
encoding_file = "data/horel_encoding.txt"  # 编码后的样本文件路径
model_save_dir = "model/chn_emotion/"
encoding_ = "utf-8-sig"


# 加载字典
def load_dict():
    with open(dict_file, "r", encoding=encoding_) as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
        return new_dict


# 根据字典对待预测评论编码
def encode_by_dict(remark, dict_encodeed):
    remark = remark.strip()
    if len(remark) <= 0:
        return []
    ret = []
    for ch in remark:  # 遍历评论每个文字
        if ch in dict_encodeed:  # 文字在字典中
            ret.append(dict_encodeed[ch])  # 对文字编码，并存入返回列表
        else:
            ret.append(dict_encodeed["<unk>"])  # 未知字符编码
    return ret  # 返回编码后的结果


# 输入测试语句、编码、执行预测
lods = []  # 待预测列表
new_dict = load_dict()  # 加载字典
lods.append(encode_by_dict("总体来说房间非常干净，卫浴设置也相当不错，交通也比较便利", new_dict))
lods.append(encode_by_dict("酒店交通方便，环境也不错，正好是我们办公地点的旁边，感觉性价比也还可以",
                           new_dict))
lods.append(encode_by_dict("设施还可以，服务人员态度也不错，交通还算便利", new_dict))
lods.append(encode_by_dict("酒店服务态度极差，设施很差", new_dict))
lods.append(encode_by_dict("我住过的最不好的酒店，以后绝不会再住了", new_dict))
lods.append(encode_by_dict("说实话，我很失望，我想这家酒店以后无论如何也不会再去了", new_dict))

# 获取每个句子的词数量
base_shape = [[len(c) for c in lods]]

# 执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

# 创建LoDTensor
tensor_words = fluid.create_lod_tensor(lods, base_shape, place)

# 加载数据
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(dirname=model_save_dir,
                                  executor=infer_exe)

# 执行预测
result = infer_exe.run(program=infer_program,
                       feed={feed_target_names[0]: tensor_words},
                       fetch_list=fetch_targets)

# 打印每个预测结果正面、负面概率
for i, r in enumerate(result[0]):
    print("负面：%f，正面：%f" % (r[0], r[1]))
