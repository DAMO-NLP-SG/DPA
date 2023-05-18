#!/bin/bash
DEVICE=0
SEED=42
((NUM=SEED-41))
CUDA_VISIBLE_DEVICES=$DEVICE python run-prompt.py --output_dir xnli-result/mixup-alpha1.2/1 \
                                        --dataset xnli \
                                        --seed $SEED \
                                        --tune_LM \
                                        --mixup_strategy hidden \
                                        --mixup_alpha 1.2 \
                                        --multi_lingual_optim \
                                        --max_train_samples 32 \
                                        --max_eval_samples 32 \
                                        --train_language en \
                                        --do_train \
                                        --do_predict \
                                        --language ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
                                        --save_strategy epoch \
                                        --evaluation_strategy epoch \
                                        --load_best_model_at_end \
                                        --metric_for_best_model eval_accuracy \
                                        --greater_is_better True \
                                        --save_total_limit 2 \
                                        --per_device_train_batch_size 2 \
                                        --per_device_eval_batch_size 8 \
                                        --gradient_accumulation_steps 4 \
                                        --learning_rate 1e-5 \
                                        --lr_scheduler_type constant \
                                        --num_train_epochs 50         
