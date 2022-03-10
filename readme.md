# causalEval
This is the source code for paper: The Invisible Hand: Understanding the Risks of Prompt-based Probing from a Causal View (ACL 2022, long paper, main conference)

## Reference
If this repository helps you, please kindly cite the following bibtext:
```
to be added
```

## Usage
To reproduce our results:
### Prepare the docker environment
```bash
to be added
```
### Download the data
```bash
to be added
```
### Run the experiments

#### 3.1 Prompt Preference Bias

```bash
bash run_prompt_variance.sh
```

Table 1的LAMA P@1结果保存在output/prompt_preference/LAMA_P文件夹下，对应到每个模型。
Table 1的其他结果存在output/prompt_preference/prompt2precion.txt文件中的最后几个大表里头，可以直接看平均结果

Figure 3和Figure 4的结果在output/prompt_preference/prompt2precion.txt中每个关系的子表里面，分别为关系P1412，P30，P140，P127

#### 3.2 Instance Verbalization Bias

```bash
bash run_instance_verb.sh
```

Figure 5的结果在output/instance_verb中，对应到每个模型。

#### 3.3 Sample Disparity Bias

```bash
bash run_sample_disparity.sh
```
Table 3的结果在output/sample_disparity中，对应到每个模型和$\gamma$的参数。

#### 3.4 Bias Elimination via Causal Intervention

```bash
bash run_causal_intervention.sh
```

Table 4的结果在output/causal_intervention/final_rank_consis_1000_20_42中