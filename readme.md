# causalEval
This is the source code for paper: Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View (ACL 2022, long paper, main conference)

## Reference
If this repository helps you, please kindly cite the following bibtext:
```
to be added
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
gdown https://drive.google.com/uc?id=1NmsEo-GGFMDvQ3s-C6L5lzpSneRAVSQv
unzip fact_data.zip
rm fact_data.zip
```
### Run the experiments

#### 3.1 Prompt Preference Bias (Table 1, Figure 3 and 4)

```bash
bash run_prompt_variance.sh
```

The results are saved in the folder "output/prompt_preference".

#### 3.2 Instance Verbalization Bias (Figure 5)

```bash
bash run_instance_verb.sh
```

The results are saved in the folder "output/instance_verb".

#### 3.3 Sample Disparity Bias (Table 3)
Prepare the pretrain corpus:
```bash
to be added
```
Pretrain and evaluate:
```bash
bash run_sample_disparity.sh
```
The script will first further pretrain four PLMs on datasets with various $\gamma$, and save the models in the folder "checkpoints".

And then evaluate all the checkpoints on test dataset, the results are saved in the folder "output/sample_disparity".

**Further Pretraining Details**

Training was performed on $8$ 40G-A100 GPUs for $3$ epochs, with maximum sequence length $512$. The batch sizes for BERT-base, BERT-large, GPT2-base, GPT2-medium are $256, 96, 128, 64$ respectively. All the models is optimized with Adam using the following parameters: $\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$ and the learning  rate is $5e-5$ with warmup ratio=$0.06$. 

#### 3.4 Bias Elimination via Causal Intervention

```bash
bash run_causal_intervention.sh
```
The results are saved in the folder "output/causal_intervention".