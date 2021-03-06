# CVPR 2022 NAS Track2：Performance Estimation 3rd Solution
> CVPR 2022 神经网络结构搜索 Track2:性能预估赛道 第三名解决方案

### 摘要
```
使用机器学习回归法及深度学习端到端学习两种方案解决该问题.
1. 机器学习回归打分法: 将任务看作回归任务, 建立回归预测模型再rank得到性能排序, 回归模型训练前使用逆误差函数将target转换为标准正态分布可以显著提升预测效果.
2. 深度学习端到端法: 将不可导的rank相关性kendall设计成可导的soft形式, 使用 1 - soft_kendall 作为loss直接训练, 模型结构使用包含稠密层的Bi-LSTM以及transformer encoder-decoder.
以上2种方法共计9次训练, 得到9个sub文件, 简单加权融合后获得最终结果即可获得线上第三名的成绩
```

## 项目结构
```
-|data
-│   ├── CVPR_2022_NAS_Track2_test.json
-│   └── CVPR_2022_NAS_Track2_train.json
-|sub
-│   ├── CVPR_2022_lgb_score.json
-│   ├── CVPR_2022_paddle_superliner_score.json
-│   ├── CVPR_2022_torch_ohe_2lstm_4logits_weight_kednall_tanh1.json
-│   ├── CVPR_2022_transformer_encoder_decoder_tanh.json
-│   ├── CVPR_2022_lstm2y_catall_tanh.json
-│   ├── CVPR_2022_lstm2y_catall_tanh_sig.json
-│   ├── CVPR_2022_lstm2y_catall_pair_sig.json
-│   ├── CVPR_2022_lstm2y_catall_tanh1_sig.json
-│   ├── CVPR_2022_ohe_lstm2y_catall_tanh1_sig.json
-│   ├── sub0.json
-│   └── sub1.json
-|model
-|loss
-|fig
-|ori
1_cvpr_lgb_score_ranker.ipynb
2_cvpr_paddle_superlinear_score_ranker.ipynb
3_cvpr_ohe_2lstm_4logits_weight_kednall_tanh1_decoder.ipynb
4_cvpr_transf_encoder_decoder_tanh.ipynb
5_cvpr_lstm2y_catall_tanh.ipynb
6_cvpr_lstm2y_catall_tanh_sig.ipynb
7_lstm2y_catall_pair_sig.ipynb
8_lstm2y_catall_tanh1_sig.ipynb
9_ohe_lstm2y_catall_tanh1_sig.ipynb
10_get_subf.ipynb
README.md
requirements.txt
```

## 一、背景介绍
神经网络结构对模型性能起着至关重要的作用，然而神经网络结构的设计依赖于专家经验，且训练及验证结构有效性的过程非常耗费算力，因此如何在不训练或少训练的情况下准确预测出各结构的性能高低，给神经网络结构设计提供自动化的参考方向十分有意义。

本任务中给出了10万个神经网络结构，其中500个结构在8个任务上进行了训练，并分别给出了各结构在8个任务上性能的相对排名（rank0-499），选手需要对剩余99500个网络结构在8个任务上的性能分别进行预测，并给出性能的相对排名（rank0-99499）。每个神经网络结构由1个37位编码表示。

任务关注不同网络结构性的性能排序，因此使用rank相关性kendall作为评估指标。

## 二、方案简介
### 2.1 网络结构打分法

网络结构打分法的思路将任务看作回归问题。将网络结构的组成看作不同的变量（如37位编码看作37个变量），对变量或变量组合进行回归模型预测打分，最终将各结构的分数进行rank获得性能高低的预测；

#### 2.1.1 LightGBM 基础打分模型
决策树是一种对特征空间进行切割，不同空间给予不同分数的模型，因此基于树的模型天然适合作为打分器。使用基于GBDT的LightGBM作为打分模型，但由于10万个模型结构各不相同，训练时过于复杂的神经网络结构组合会导致严重的过拟合问题，直接建模a榜分数只有0.7左右，降低模型中神经网络结构组合的复杂程度，简单将树的叶子节点数调整为2便可轻松到达0.78+。

#### 2.1.2 One-Hot Linear 基础打分模型
由于打分模型中的神经网络组合过于复杂会导致严重的过拟合，且假设树模型进一步降低复杂度，退化为单叉树，此时作为加法模型的GBDT将与做了One-Hot的linear有着基本一致的表达形式。因此进一步使用做了One-Hot，8个目标同时建模的Linear作为打分模型，该模型亦可轻松到达0.78+。

#### 2.1.3 提升-超级打分模型
任务中训练和预测的标签均为性能的rank等级，相邻标签间的距离均为1，可以看作均匀分布。但实际上，模型性能高低的距离并非等距，而是大部分模型性能差不多，少部分模型性能很好或很差，整体可看作高斯分布。因此使用逆误差函数(erfinv)将标签变换为高斯分布，使用变换后的取值进行打分回归模型的训练，使用该变换后基础的打分模型可到达0.793+的分数。

```python
# 使用逆误差函数将target转换为高斯分布

from scipy.special import erfinv
def erfinv_trans(train_df, c):
    train_y = train_df[c]
    mmin = np.min(train_y) + 1
    mmax = np.max(train_y) + 1
    train_y = np.sqrt(2) * erfinv(2 * (train_y + mmin) / (mmin + mmax) - 1)
    train_df[c + '_trans_y'] = train_y
```

标签的原始分布:
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/target_ori.png" width="500" height="400">
</div>

标签逆误差函数转换后的分布:
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/target_erfinv_trans.png" width="500" height="400">
</div>

```
高斯分布下的LightGBM打分模型：
1_cvpr_lgb_score_ranker.ipynb
高斯分布下One-Hot,8目标同时建模的Super Linear打分模型：
2_cvpr_paddle_superlinear_score_ranker.ipynb
使用的打分模型代码可直接运行，模型存至./model，结果存至./sub。
```


### 2.2 Rank Loss端到端法
基于回归的打分法旨在降低预测分数和实际分数的差异, 但在神经网络结构搜索中, 准确预测出不同网络结构的性能相对高低比降低分数的差异更为重要, 比如两个结构的实际预测精度分别是0.6和0.7,在回归任务中预测结果0.7和0.8的loss和预测为0.7和0.6的loss完全相同, 但显然后一个预测颠倒了两个结构性能的高低排序, 属于失败的预测. 因此基于回归的打分法实际上并不能很好的解决该问题，而深度学习的一大优势便是可以端到端学习，可使用端到端的思路将神经网络结构编码至高维空间，再直接解码为排序高低关系。

#### 2.2.1 rank loss设计
rank关系是一类不可导的计算过程，评价任务性能的kendall也是其中之一，在本任务中将不可导的kendall设计为可导的形式尤为重要，我的解决方案是将不可导的地方做soft处理, 从而获得可导的 soft_kendall, 最后使用1-soft_kendall作为loss进行训练。

<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/kendall_exp.png" width="793" height="72">
</div>
    
```
kendall的计算过程如下：
1. 以表1为例，首先将预测向量按实际等级升序的顺序进行排序；
2. 将预测的第一个数字2依次与右边的每个数字进行比较（共9次），大于第一个数字2则记+1分，小于2则-1分，总分为-1+8=7分；
3. 再将第二个数字1依次与右边的8个数字进行比较，得到分数8；
4. 以此类推，得到9个分数:7,8,7,2,5,4,1,0,1,总分35分；
5. 如果所有的分数都为最大分数，9,8,7,6,5,4,3,2,1，则最大总分为45分；
6. kendall = 实际得分/最大总分，即35/45 = 0.778;
```

这个过程中的排序及计分过程均不可导，但排序操作可以仅针对真实标签，并不影响预测结果的可导性与反向传播，因此仅需解决计分过程的可导性。其中计分函数可以看作sign函数，比较结果>0 取1，比较结果<0 取-1，因此可使用tanh，erf，sigmoid，softsign等类似的可导函对计分过程进行替换，便可以实现可导的soft kendall 

不同soft函数及其梯度如下,相关代码在./fig/soft&grad.ipynb
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/softcurves.png" width="800" height="480">
</div>
    
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/grad_of_softcurves.png" width="800" height="480">
</div>
    

以tanh为例的soft kendall代码如下，

```python
# kendall的手动python实现, 将其中的sign变为tanh即可实现基于tanh的soft kendall
def kendall(actuals, preds):
    sa = actuals.argsort()[0]
    tmp = preds.index_select(1, sa.int())[0]
    score = torch.cat([((tmp[i:] - tmp[i-1])).sign() for i in range(1, tmp.shape[0])]).sum()
    score1 = score / sum(list(range(1, len(tmp))))
    return score1

# 下面是1-soft_kendall作为loss的示例代码,分别为torch版本以及paddle版本

# 使用tanh的soft kendall,torch版本
class CVPRLoss_tanh(nn.Module):
    # 1-soft_kendall tanh
    def __call__(self, pred, y):
        return 1-torch.cat(
            [self.get_score(y[:,i].reshape(1,-1), 
                           pred[:,i].reshape(1,-1)).reshape(1,-1) for i in range(y.shape[1])]
        ).reshape(1, -1)
    
    def get_score(self, actuals, preds):
        sa = actuals.argsort()[0]
        tmp = preds.index_select(1, sa.int())[0]
        score = torch.cat([((tmp[i:]-tmp[i-1])).tanh() for i in range(1, tmp.shape[0])]).sum()
        score1 = score/sum(list(range(1,len(tmp))))
        return score1

# 使用tanh的soft kendall, paddle版本
class CVPRLoss_tanh(nn.Layer):
    # 1 - soft_kendall tanh
    def __init__(self):
        super(CVPRLoss_tanh, self).__init__()
        
    def forward(self, pred, y):
        return 1-paddle.concat(
            [self.get_score(y[:,i], pred[:,i]) for i in range(y.shape[1])]
        )
        
    def get_score(self, actuals, preds):
        sa = actuals.argsort()
        tmp = preds.index_select(index=sa)
        score = paddle.concat([((tmp[i:]-tmp[i-1])).tanh() for i in range(1, tmp.shape[0])]).sum()
        score1 = score/sum(list(range(1,len(tmp))))
        return score1

./loss文件夹下提供了建模过程中曾使用到的loss
其中基于tanh的矩阵版soft kendall性能最好
CVPRLoss:  使用1-pearson相关性作为loss;
CVPRLoss1: 使用1-spearman rank相关性作为loss;
CVPRLoss_tanh: 使用tanh改写的1-kendall rank相关性作为loss;
CVPRLoss_erf: 使用erf改写的1-kendall rank相关性作为loss;
CVPRLoss_sigmoid: 使用sigmoid改写的1-kendall rank相关性作为loss;
CVPRLoss_softsign: 使用softsign改写的1-kendall rank相关性作为loss;
CVPRLoss_tanh1: 使用tanh改写的1-kendall rank相关性作为loss，矩阵写法;
CVPRLoss_pair： 使用huawei-noah定义的一种rank loss作为loss；
```

#### 2.2.2 模型结构
rank loss 建模时将输入看作长度为37的序列, 模型结构基于 包含稠密层的Bi-LSTM 以及 transformer 编码器解码器, 具体使用了以下三种结构:
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/model1.png" width="356" height="782">
</div>

<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/model2.png" width="454" height="684">
</div>
    
<div align=center>
<img src="https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/fig/model3.png" width="583" height="768">
</div>
    

```
使用结构3的模型:
3_cvpr_ohe_2lstm_4logits_weight_kednall_tanh1_decoder.ipynb
使用结构2的模型:
4_cvpr_transf_encoder_decoder_tanh.ipynb
使用结构1的模型:
5_cvpr_lstm2y_catall_tanh.ipynb
6_cvpr_lstm2y_catall_tanh_sig.ipynb
7_lstm2y_catall_pair_sig.ipynb
8_lstm2y_catall_tanh1_sig.ipynb
9_ohe_lstm2y_catall_tanh1_sig.ipynb

其中,5-9结构相同,但训练过程略有区别:
5_cvpr_lstm2y_catall_tanh.ipynb 使用 tanh soft kendall loss, 取8目标整体最优;
6_cvpr_lstm2y_catall_tanh_sig.ipynb 使用 tanh soft kendall loss, 取8目标交叉验证时单目标最优;
7_lstm2y_catall_pair_sig.ipynb 为 矩阵写法的tanh soft kendall loss,取8目标交叉验证时单目标最优;
8_lstm2y_catall_tanh1_sig.ipynb 为 使用huawei-noah的pair loss,取8目标交叉验证时单目标最优;
9_ohe_lstm2y_catall_tanh1_sig.ipynb 为 矩阵写法的tanh soft kendall loss,取8目标交叉验证时单目标最优, 输入为one-hot的序列, 500*37*93

基于rank loss的各模型分数介于0.792-0.795不等.

打分法模型及rank loss模型均无任何手工特征, 特征工程角度仅使用到编码的one-hot.
```

### 2.3 模型融合
10_get_subf.ipynb 为获取最终提交的模型融合代码, 代码输出2个提交, 2个提交均使用上述2个思路,5种模型,9次训练得到的9个sub进行融合。
区别在于第一个提交对各目标的最优模型权重更大, 第二个提交按 最优:rank_loss:打分 = 4:4:2 进行权重分配。

子模型中打分模型最高可做到0.793+, 但融合中主要用于提高模型的差异性, 保证泛化能力, 因此分配了更低的权重, 并且为了保证没有过度拟合榜单, 最终使用的打分模型并未参考榜单结果使用最高分模型, 而是提取了0.791+和0.792+的两个前中期打分模型。

最终的模型融合仅在最后一天进行了4次, 并未做大量数据试验, 不一定是最优权重。

#### 使用方式
```
分别运行
1_cvpr_lgb_score_ranker.ipynb
2_cvpr_paddle_superlinear_score_ranker.ipynb
3_cvpr_ohe_2lstm_4logits_weight_kednall_tanh1_decoder.ipynb
4_cvpr_transf_encoder_decoder_tanh.ipynb
5_cvpr_lstm2y_catall_tanh.ipynb
6_cvpr_lstm2y_catall_tanh_sig.ipynb
7_lstm2y_catall_pair_sig.ipynb
8_lstm2y_catall_tanh1_sig.ipynb
9_ohe_lstm2y_catall_tanh1_sig.ipynb
得到9个模型及其预测结果;
然后运行10_get_subf.ipynb 得到最终提交的2个输出
其中模型存在./model；预测结果存在./sub

>注：
>1. 其中1，2打分模型非常小，总计不超过10Mb;
>2. 端到端模型单个模型90MB-450M不等，但任务中使用了5折训练，并在部分训练中(以_sig结尾的代码)分别保存了8个target的5折最优模型（即一次训练保存了5*8=40个模型文件），因此端到端虽然只进行了7次训练, 但8个目标的5折合计保存了210个模型文件,得到的模型文件总计70Gb左右;
>3. 以上代码均为剔除了冗余, 无意义注释, 修改了非顺序执行的复现代码, 原始代码及原始提交见./ori, 若需参考原始代码, 提前阅读./ori下的readme文件
```

#### 参考文献

[1] Li Z, Xi T, Deng J, et al. Gp-nas: Gaussian process based neural architecture search[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 11933-11942.

[2] Xu Y, Wang Y, Han K, et al. Renas: Relativistic evaluation of neural architecture search[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 4411-4420.

[3] Blondel M, Teboul O, Berthet Q, et al. Fast differentiable sorting and ranking[C]//International Conference on Machine Learning. PMLR, 2020: 950-959.
