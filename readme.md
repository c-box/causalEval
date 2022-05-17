# causalEval
This is the source code for paper: Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View (ACL 2022, long paper, main conference)

## Reference
If this repository helps you, please kindly cite the following bibtext:
```
@inproceedings{cao-etal-2022-prompt,
    title = "Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View",
    author = "Cao, Boxi  and
      Lin, Hongyu  and
      Han, Xianpei  and
      Liu, Fangchao  and
      Sun, Le",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.398",
    pages = "5796--5808",
    abstract = "Prompt-based probing has been widely used in evaluating the abilities of pretrained language models (PLMs). Unfortunately, recent studies have discovered such an evaluation may be inaccurate, inconsistent and unreliable. Furthermore, the lack of understanding its inner workings, combined with its wide applicability, has the potential to lead to unforeseen risks for evaluating and applying PLMs in real-world applications. To discover, understand and quantify the risks, this paper investigates the prompt-based probing from a causal view, highlights three critical biases which could induce biased results and conclusions, and proposes to conduct debiasing via causal intervention. This paper provides valuable insights for the design of unbiased datasets, better probing frameworks and more reliable evaluations of pretrained language models. Furthermore, our conclusions also echo that we need to rethink the criteria for identifying better pretrained language models.",
}
```

## Usage
To reproduce our results:

### Prepare the environment
```bash
conda create --name causaleval python=3.8
pip install -r requirements.txt
```

### Download the data
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1ncag639r6zow9OXNn463bd2OrJ7HuJzv
unzip fact_data.zip
rm fact_data.zip
```
### Run the experiments

#### 3.1 Prompt Preference Bias (Table 1, Figure 3 and 4)

```bash
bash run_prompt_variance.sh
```

The results are saved in the folder "output/prompt_preference".
The relations in Figure 3 and 4 are: P1412，P30，P140，P127.

#### 3.2 Instance Verbalization Bias (Figure 5)

```bash
bash run_instance_verb.sh
```

The results are saved in the folder "output/instance_verb".

#### 3.3 Sample Disparity Bias (Table 3)
Prepare the pretrain corpus:
```bash
gdown https://drive.google.com/uc?id=1N47E2UQ5JYzJQaaeVDP7hUErDuKoejGp
unzip pretrain_data.zip
rm pretrain_data.zip
```
Pretrain and evaluate:
```bash
bash run_sample_disparity.sh
```
The script will first further pretrain four PLMs on datasets with various $\gamma$, and save the models in the folder "checkpoints".

Then evaluate all the checkpoints on test dataset, the results are saved in the folder "output/sample_disparity".

**Further Pretraining Details**

Training was performed on 8 40G-A100 GPUs for 3 epochs, with maximum sequence length 512. The batch sizes for BERT-base, BERT-large, GPT2-base, GPT2-medium are 256, 96, 128, 64 respectively. All the models is optimized with Adam using the following parameters: $\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$ and the learning  rate is $5e-5$ with warmup ratio=$0.06$. 

#### 3.4 Bias Elimination via Causal Intervention

```bash
bash run_causal_intervention.sh
```
The results are saved in the folder "output/causal_intervention".

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
