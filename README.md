# TextSimilarity
Text similarity projects.      To be continued...         

## 1. [短文本相似性](短文本相似性/)
   
   模型运行环境：
   > - python 2.7
   > - tensorflow-gpu >= 1.4.0
   > - keras >= 2.1.0
   
---
   
   模型思路：[[code]]()
   > 1. pretrain + train: 原数据上pretrain，再使用data argumentation后的数据训练，充分利用有限数据。
   > 2. embedding dropout: 在feature维度和word vec维度上分别进行dropout，避免过拟合，提高泛化能力。
   > 3. graph feature argumentation: 使用图建立文本间相似性的传递。[[code]](短文本相似性/code/graph_feature_generate.ipynb)
   > 4. sen1与sen2共享编码器: 使得分析对象处于同一空间中。
   > 5. 三种编码结构: 
         - 使用lstm学习序列时序信息；
         - 使用gru堆叠在lstm之上，学习更高层抽象的信息；
         - 使用多个不同大小的dilation convolution学习局部的关联信息。
        模型不具备对称性，但是句子对有，所以交换输入顺序，predict两次，取均值。
   > 6. 相似性计算方法:
         - 使用编码后feature vec的点积（lstm与gru层输出）、逐元素差和逐元素相乘（concat层输出）进行相似度计算，不同vec进行concat和
           batch normalization
         - 然后接多层fully connected layers计算cross entropy loss
         
 ---
 
   模型待改进：
   > 1. 将字vec和词vec特征共同输入；
   > 2. embedding层可以使用类似scheduled sampling的方法，整合开放训练embedding带来的拟合效果和不开放embedding带来的更好的泛化效果（前提是
   使用了预训练的embedding）
