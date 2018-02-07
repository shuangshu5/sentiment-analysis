# sentiment-analysis
通过CNN网络对京东数码产品的用户评论进行情感建模

## 环境要求
- Python 2.7
- Tensorflow 1.0
- numpy
- jieba

## Evaluating

```bash
python eval.py
```

## Training

```bash
python train.py
```

## 效果演示

输入:

```bash
这电池质量好差
```

输出:

```bash
情感倾向: 负面情感   置信度: 0.999433
```
