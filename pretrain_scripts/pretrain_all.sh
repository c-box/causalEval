DEVICE=0

bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 0.0 8 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 0.2 8 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 0.4 8 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 0.6 8 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 0.8 8 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2-medium 1.0 8 3

bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.0 16 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.0 2 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.2 16 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.4 16 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.6 16 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 0.8 16 3
bash pretrain_scripts/run_clm.sh $DEVICE gpt2 1.0 16 3

bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 0.0 32 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 0.2 32 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 0.4 32 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 0.6 32 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 0.8 32 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-base-cased 1.0 32 1 3

bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 0.0 12 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 0.2 12 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 0.4 12 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 0.6 12 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 0.8 12 1 3
bash pretrain_scripts/run_mlm.sh $DEVICE bert-large-cased 1.0 12 1 3