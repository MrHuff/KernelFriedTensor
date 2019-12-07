for i in {1..5}
do
  python run_job_script.py \
    --batch_size_a 0.005 \
    --batch_size_b 0.1 \
    --max_lr 1e-1 \
    --max_R 25 \
    --PATH ./tensor_data/ \
    --reg_para_a 10 \
    --reg_para_b 50 \
    --fp_16 False \
    --fused True \
    --save_path private_job_arch_0 \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 1 \
    --side_info_order 0 1 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual True \
    --init_max 0.01 \
    --delete_side_info 0 1 \
    --old_setup False \
    --task reg
done

