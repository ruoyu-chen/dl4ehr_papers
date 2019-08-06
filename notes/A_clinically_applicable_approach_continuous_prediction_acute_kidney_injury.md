# A clinically applicable approach to continuous prediction of future acute kidney injury

题目：一种临床可应用的持续预测未来急性肾损伤的方法  
摘要：The early prediction of deterioration could have an important role in supporting healthcare professionals, as an estimated 11% of deaths in hospital follow a failure to promptly recognize and treat deteriorating patients1. To achieve this goal requires predictions of patient risk that are continuously updated and accurate, and delivered at an individual level with sufficient context and enough time to act. Here we develop a deep learning approach for the continuous risk prediction of future deterioration in patients, building on recent work that models adverse events from electronic health records2–17 and using acute kidney injury—a common and potentially life-threatening condition18—as an exemplar. Our model was developed on a large, longitudinal dataset of electronic health records that cover diverse clinical environments, comprising 703,782 adult patients across 172 inpatient and 1,062 outpatient sites. Our model predicts 55.8% of all inpatient episodes of acute kidney injury, and 90.2% of all acute kidney injuries that required subsequent administration of dialysis, with a lead time of up to 48 h and a ratio of 2 false alerts for every true alert. In addition to predicting future acute kidney injury, our model provides confidence assessments and a list of the clinical features that are most salient to each prediction, alongside predicted future trajectories for clinically relevant blood tests9. Although the recognition and prompt treatment of acute kidney injury is known to be challenging, our approach may offer opportunities for identifying patients at risk within a time window that enables early treatment.  

## 1. 概述  
文中提出的系统是基于循环神经网络（RNN）的，以单个EHR记录为单位，一次处理一个记录，在RNN内部建立起对相关信息的内部记忆。  
```
Our proposed system is a recurrent neural network that operates sequentially over individual electronic health records, processing the data one step at a time and building an internal memory that keeps track of relevant information seen up to that point. 
```
数据集中的独立记录的个数大约为60亿，包括了**62万个特征**。  
``` 
The total number of independent entries in the dataset was approximately 6 billion, includ- ing 620,000 features. 
```
数据集被划分为4个部分：训练集（80%），验证集（5%），校准集（5%）以及测试集（10%）。  
```Patients were randomized across training (80%), validation (5%), calibration (5%) and test (10%) sets```
