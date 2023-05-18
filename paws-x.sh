#!/bin/bash
DEVICE=0
SEED=42
CUDA_VISIBLE_DEVICES=$DEVICE python run_pawsx.py --output_dir pa-result/FT/128shots/1 \
                                        --seed $SEED \
                                        --max_train_samples 128 \
                                        --max_eval_samples 128 \
                                        --train_language en \
                                        --overwrite_output_dir \
                                        --do_train \
                                        --do_predict \
                                        --language de,en,es,fr,zh,ja,ko \
                                        --save_strategy epoch \
                                        --evaluation_strategy epoch \
                                        --load_best_model_at_end \
                                        --metric_for_best_model eval_accuracy \
                                        --greater_is_better True \
                                        --save_total_limit 2 \
                                        --per_device_train_batch_size 8 \
                                        --per_device_eval_batch_size 32 \
                                        --gradient_accumulation_steps 1 \
                                        --learning_rate 1e-5 \
                                        --lr_scheduler_type constant \
                                        --num_train_epochs 50         
