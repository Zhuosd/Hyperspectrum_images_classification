## python复现

### 复现工具版本
```
python                3.7
numpy                1.21.6
scipy                1.7.3
libsvm               3.25
scikit-learn         1.0.2 (可选)
PyWavelets           1.3.0     小波变换
scikit-image         0.19.3 
```
* 复现工具中，所有均可以通过pip进行安装，libsvm通过pip安装后，还需进一步操作，查看链接 https://blog.csdn.net/jeryjeryjery/article/details/72628255 https://blog.csdn.net/weixin_32557949/article/details/112989695
***

### 文件说明
* libmat为原始mat数据存放位置
* libsvm 存在一个当前版本的pip安装包，还有对应svm的源文件（pip安装后，可以不用到这些部分）
* mfun 为原始主文件中调用的函数复现存放位置，调用也是基于此
* GraphcutMet 为GitHub中clone的源文件（目前尚未涉及python调用）

# 实现（除了graphcut函数）
* Demo_SVM_MRF_ZaoyuanV0817
* Demo_DRF_EM_DFC3rd（大部分）

# 遇到的问题
* lorsal 函数中有线性代数计算特征值和特征向量，这个导致函数输出结果不一致（exp指数运算产生inf溢出）
* BP_message 中大维度（三十五万循环）导致计算速度很慢，大致13~15分钟，循环中浮点数矩阵乘法和除法运算导致最终输出的概率值 有0.05~0.08 误差值
