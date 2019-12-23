for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.05 \
    --max_lr 1e-1 \
    --max_R 30 \
    --PATH ./public_data/ \
    --reg_para_a 1e-5 \
    --reg_para_b 1 \
    --fp_16 False \
    --fused True \
    --save_path public_job_arch_0_dual_bayesian_multivariate \
    --seed $i \
    --bayesian True \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 0 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual True \
    --multivariate True \
    --init_max 0.1 \
    --task reg > public_job_$i.out &
  sleep 10
done

