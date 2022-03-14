# python prompt_variance.py 

DEVICE=2

# python evaluate_model.py --model-name bert-large-cased --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 64 --cuda-device $DEVICE

# python evaluate_model.py --model-name roberta-large --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 64 --cuda-device $DEVICE

# python evaluate_model.py --model-name gpt2-xl --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 32 --cuda-device $DEVICE

# python evaluate_model.py --model-name bart-large --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 32 --cuda-device $DEVICE

# python evaluate_model.py --model-name bert-base-cased --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 64 --cuda-device $DEVICE

# python evaluate_model.py --model-name gpt2 --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 64 --cuda-device $DEVICE

python evaluate_model.py --model-name gpt2-medium --task data_evaluation --out-dir output/prompt_preference/LAMA_P --batch-size 64 --cuda-device $DEVICE