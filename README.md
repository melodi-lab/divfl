# Diverse Client Selection for Federated Learning via Submodular Maximization

## Code for ICLR 2022 paper:

<b>Title</b>: <i>Diverse Client Selection for Federated Learning via Submodular Maximization</i> <a href="https://openreview.net/pdf?id=nwKXyFvaUm">[pdf]</a> <a href="https://iclr.cc/virtual/2022/poster/7047">[presentation]</a>\
<b>Authors</b>: Ravikumar Balakrishnan* (Intel Labs), Tian Li* (CMU), Tianyi Zhou* (UW), Nageen Himayat (Intel Labs), Virginia Smith (CMU) and Jeff Bilmes (UW)\
<b>Institutes</b>: Intel Labs, Carnegie Mellon University, University of Washington

<pre>
@inproceedings{
balakrishnan2022diverse,
title={Diverse Client Selection for Federated Learning via Submodular Maximization},
author={Ravikumar Balakrishnan and Tian Li and Tianyi Zhou and Nageen Himayat and Virginia Smith and Jeff Bilmes},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nwKXyFvaUm}
}</pre>


<b>Abstract</b>\
In every communication round of federated learning, a random subset of clients communicate their model updates back to the server which then aggregates them all. The optimal size of this subset is not known and several studies have shown that typically random selection does not perform very well in terms of convergence, learning efficiency and fairness. We, in this paper, propose to select a small diverse subset of clients, namely those carrying representative gradient information, and we transmit only these updates to the server. Our aim is for updating via only a subset to approximate updating via aggregating all client information. We achieve this by choosing a subset that maximizes a submodular facility location function defined over gradient space. We introduce “federated averaging with diverse client selection (DivFL)”. We provide a thorough analysis of its convergence in the heterogeneous setting and apply it both to synthetic and to real datasets. Empirical results show several benefits to our approach including improved learning efficiency, faster convergence and also more uniform (i.e., fair) performance across clients. We further show a communication-efficient version of DivFL that can still outperform baselines on the above metrics.
