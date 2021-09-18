export PYTHONPATH=$PYTHONPATH:../../../../
log_dir=dp2_pp2_mp2
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --gpus "7" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 10000\
    --save_steps 100000\
    --decay_steps 320000\
    --device gpu\
    --eval_freq 1000\
    --warmup_rate 0.01\
    --scale_loss 32768\
    --global_batch_size 8\
    --micro_batch_size 2\
    --dp_degree 1\
    --mp_degree 1\
    --pp_degree 1\
    --use_pure_fp16 False\
    --use_recompute False

