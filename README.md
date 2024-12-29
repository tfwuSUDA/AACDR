# AACDR
**—— Integrating Graph Isomorphism Networks and Asymmetric Adversarial Domain Adaptation for Cancer Drug Response Prediction**

Predicting cancer drug response is critical for the development of personalized treatments, as it helps identify optimal therapeutic options for cancer patients. Due to the limited scale of clinical trial data, previous research has focused on preclinical data, often overlooking the distribution discrepancies between these data. To address this issue, a cancer drug response prediction method with asymmetric adversarial domain adaptation (AACDR) is introduced. This approach addresses the limitations of conventional adversarial domain adaptation methods by pushing distribution of target domain towards that of source, achieving greater accuracy and stability in predictions of target task. Drug feature extraction is enhanced by using graph isomorphism networks, enabling a more comprehensive data representation. Experiments on cancer patient datasets demonstrate effective knowledge transfer from cell lines to patients, outperforming existing methods. Further validation on Patient-Derived Xenograft (PDX) dataset highlights the generalizability of AACDR across various distributional discrepancies. Additionally, testing on Copy Number Variation (CNV) dataset demonstrates its adaptability to different cancer representation methods. The model not only accurately predicts therapeutic results for real clinical records, but also recommends potential therapeutic options for specific cancer patients, supported by relevant studies, underscoring its practical importance in delivering personalized treatment strategies.
**This source code was tested on the basic environment with `conda==24.5.0` and `cuda==11.8`**

## Environment Reproduce
- In order to get AACDR, you need to clone this repo:
  ```
    git clone https://github.com/tfwuSUDA/AACDR.git
    cd AACDR
    conda env create -f environment.yaml
  ```

## File Description
- data: include the original data of Expr, CNV and PDX datasets.
- code: source code of AACDR.
- supplymentarymaterials:
    - 100_random_initializations.txt: the results of 100 random initializations and training of AACDR.
    - 28DifferentHyperparameters.xlsx: the results of 28 different hyperparameters of AACDR and PANCD.
    - supplymentary material of AACDR.pdf: other supplymentary materials.

## Make Dataset
- In order to make Expr, CNV and PDX datasets, you need to unzip the file:AACDR/data/TCGA/TCGA_unlabel_expr_702_01A.csv.gz and run this command to make dataset:
  ```
    python uitils.py
  ```

## Run & Test
please use the following command to train AACDR(100 random initialization):
  ```
    python main.py --description Reproduce --id 0 
  ```

**contact**
Yi Zhang: 20235227030@stu.suda.edu.cn
Tingfang Wu: tfwu@suda.edu.cn
