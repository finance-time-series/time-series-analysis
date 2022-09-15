### Abstract

* 금융 시계열 데이터의 예측 및 이상탐지 목표로 널리 알려진 딥러닝 모델뿐만 아니라 최근에 주목받고 있는 다양한 모델들을 적용하여 예측 및 이상탐지 실험
* 지급결제 데이터([한국은행 경제통계시스템](https://ecos.bok.or.kr)), 계좌 입출금 내역([체코은행](https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions), [Kaggle](https://www.kaggle.com/datasets/apoorvwatsky/bank-transaction-data?datasetId=215646)), ATM 입출금 내역([Kaggle](https://www.kaggle.com/datasets/nitsbat/data-of-atm-transaction-of-xyz-bank)) 등의 금융 데이터를 대상으로 실험

--------------------------------------------------------------------------------

### Time-series forecasting model

* **[RNN]** Recurrent neural network
* **[LSTM]** Long-short term memory
* **[GRU]** from Chung, Junyoung, et al.: [Empirical evaluation of gated recurrent neural networks on sequence modeling](
https://doi.org/10.48550/arXiv.1412.3555) (NIPS 2014)
* **[NCDE]** from Kidger, Patrick, et al.: [Neural controlled differential equations for irregular time series](
https://doi.org/10.48550/arXiv.2005.08926
) (NeurIPS 2020)

--------------------------------------------------------------------------------

### Anomaly detection model
* **[LOF]** from Breunig, Markus M., et al.: [LOF: identifying density-based local outliers](https://doi.org/10.1145/335191.335388) (SIGMOD 2000)
* **[Isolation Forest]** from Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.: [Isolation forest](10.1109/ICDM.2008.17) (ICDM 2008) 
* **[Anomaly Transformer]** from Xu, Jiehui, et al.: [Anomaly transformer: Time series anomaly detection with association discrepancy](
https://doi.org/10.48550/arXiv.2110.02642) (ICLR 2022)

