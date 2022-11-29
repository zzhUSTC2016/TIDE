# TIDE

This is the pytorch and C++ implementation of our TKDE 2023 paper:

> Zhao Z, Chen J, Zhou S, et al. Popularity bias is not always evil: Disentangling benign and harmful bias for recommendation[J]. IEEE Transactions on Knowledge and Data Engineering, 2022. [Paper link](https://ieeexplore.ieee.org/document/9935285/)

# Introduction

We argue that not *all popularity bias is evil*. Popularity bias not only results from conformity but also item quality, which is usually ignored by existing methods. Some items exhibit higher popularity as they have intrinsic better property. Blindly removing the popularity bias would lose such important signal, and further deteriorate model performance. 

To sufficiently exploit such important information for recommendation, it is essential to disentangle the benign popularity bias caused by item quality from the harmful popularity bias caused by conformity. Although important, it is quite challenging as we lack an explicit signal to differentiate the two factors of popularity bias. 

In this paper, we propose to leverage temporal information as the two factors exhibit quite different patterns along the time: item quality revealing item inherent property is stable and static while conformity that depends on items' recent clicks is highly time-sensitive. Correspondingly, we further propose a novel Time-aware DisEntangled framework (TIDE), where a click is generated from three components namely the static item quality, the dynamic conformity effect, as well as the user-item matching score returned by any recommendation model.

# Dataset



# An example to run TIDE



# Results

