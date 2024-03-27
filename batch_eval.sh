num_iterations=2
# Base directory for output
base_output_dir="1e6"
for i in $(seq 1 $num_iterations); do
	output_dir="${base_output_dir}/${i}"

	accelerate launch --num_processes=8 --main_process_port=29501 batch_eval.py \
	--model_name_or_path "/data/trained_models/mediqa_training_only_1e6_new/checkpoint-76" \
	--output_dir "$output_dir" \
	--add_special_tokens False \
	--append_concat_token False \
	--splits "train_sft,test_sft" \
	--max_seq_len 4096 \
	--num_train_epochs 3 \
	--logging_steps 5 \
	--log_level "info" \
	--logging_strategy "steps" \
	--evaluation_strategy "epoch" \
	--save_strategy "epoch" \
	--push_to_hub False \
	--hub_private_repo False \
	--bf16 True \
	--packing True \
	--learning_rate 1.41e-5 \
	--lr_scheduler_type "cosine" \
	--weight_decay 0.0 \
	--warmup_ratio 0.1 \
	--max_grad_norm 1.0 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--gradient_accumulation_steps 4 \
	--gradient_checkpointing True \
	--use_reentrant False \
	--dataset_text_field "content" \
	--use_flash_attn False
	#--seed ${i} \
done
