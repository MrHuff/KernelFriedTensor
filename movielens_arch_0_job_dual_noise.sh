for i in {1..5}
do
  taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.10 \
    --max_lr 1e-1 \
    --max_R 12 \
    --PATH ./public_movielens_data/ \
    --reg_para_a 0 \
    --reg_para_b 1e-3 \
    --fp_16 False \
    --fused True \
    --save_path movielens_job_arch_0_dual_noise \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual True \
    --init_max 1e-1 \
    --mu_a 0 \
    --mu_b 1e-2 \
    --sigma_a 0.5 \
    --sigma_b 3 \
    --special_mode 2 \
    --task reg > movielens_job_$i.out &
  sleep 10
done

