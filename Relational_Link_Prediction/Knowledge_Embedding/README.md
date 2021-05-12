# Knowledge Embedding

## Benchmark

- [x] FB15k, FB15k-237
- [x] WN18, WN18RR

## Model

- [x] TransE
- [x] RotatE
- [x] DistMult
- [x] ComplEx
- [x] TuckER
- [x] ConvE
- [x] RGCN (PyG)

## Feature
- [x] 1vN, kvN, NS sampling strategy
- [x] NegativeSamplingLoss, LabelSmoothingCELoss, LabelSmoothingBCELoss

## Evaluation Setting
 

### Standard

At train time, use $(i, j, k)$.

At test time, use $(i, j, ?)$ for right hand sides and $(?, j, k)$ for left hand sides$.

Where $P$ is the number of relations.

### Reciprocal

**ref**: Canonical Tensor Decomposition for Knowledge Base Completion. ICML. 2018

At train time, use $(i, j, k)$ and $ (k, j + P, i) $.

At test time, use $(i, j, ?)$ for right hand sides and $(k, j + P, ?)$ for left hand sides.

Where $P$ is the number of relations.

### Experiments

|Model|Training|Evaluation|MRR|Hits@1|Hits@3|Hits@10|
|---|---|---|---|---|---|---|
|ComplEx|Standard (corrput lhs and rhs)|Standard|0.3398|0.2490|0.3717|0.5241|
|ComplEx|Reciprocal (corrput lhs and rhs)|Standard|0.3384|0.2470|0.3728|0.5238|
|ComplEx|Reciprocal (corrput lhs and rhs)|Reciprocal|0.3403|0.2488|0.3732|0.5236|
|ComplEx|Reciprocal (corrput rhs)|Reciprocal|0.3435|0.2515|0.3771|0.5274|
