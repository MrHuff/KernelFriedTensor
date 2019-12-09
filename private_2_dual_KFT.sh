for i in {1..5}
do
  python run_job_script.py \
    --batch_size_a 0.005 \
    --batch_size_b 0.1 \
    --max_lr 1e0 \
    --max_R 15 \
    --PATH ./tensor_data/ \
    --reg_para_a 0 \
    --reg_para_b 1e-3 \
    --fp_16 False \
    --fused True \
    --save_path private_job_arch_1_KFT \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 1 \
    --side_info_order 0 1 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual True \
    --init_max 0.1 \
    --delete_side_info 0 1 \
    --old_setup False \
    --task reg
done

