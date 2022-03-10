DEVICE=4

bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 0.0 8 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 0.2 8 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 0.4 8 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 0.6 8 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 0.8 8 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-medium 1.0 8 3


bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 0.0 12 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 0.2 12 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 0.4 12 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 0.6 12 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 0.8 12 3
bash pretrain_scripts/evaluate_gpt.sh $DEVICE gpt2-large 1.0 12 3


bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 0.0 32 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 0.2 32 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 0.4 32 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 0.6 32 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 0.8 32 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-base-cased 1.0 32 1 3


bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 0.0 12 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 0.2 12 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 0.4 12 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 0.6 12 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 0.8 12 1 3
bash pretrain_scripts/evaluate_bert.sh $DEVICE bert-large-cased 1.0 12 1 3