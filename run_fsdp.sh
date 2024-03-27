accelerate launch --config_file "configs/fsdp_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "final_new_Mistral-7B-v0.1" \
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
--learning_rate 2e-6 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--max_grad_norm 1.0 \
--output_dir "test" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn False