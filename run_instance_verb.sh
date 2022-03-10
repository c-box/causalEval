DEVICE=3

python cal_mention_consis.py --model-name bert-large-cased --cuda-device $DEVICE

python cal_mention_consis.py --model-name roberta-large --cuda-device $DEVICE

python cal_mention_consis.py --model-name gpt2-xl --cuda-device $DEVICE

python cal_mention_consis.py --model-name bart-large --cuda-device $DEVICE