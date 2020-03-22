for i in {1..5}
do
  taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.05 \
    --max_lr 5e-2 \
    --max_R 20 \
    --PATH ./public_data/ \
    --reg_para_a 1e-3 \
    --reg_para_b 1 \
    --save_path public_job_arch_0_dual_bayesian_multivariate_ls \
    --seed $i \
    --bayesian True \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 0 2 \
    --temporal_tag 2 \
    --latent_scale True \
    --dual True \
    --multivariate True \
    --init_max 1.0 \
    --mu_a 0 \
    --mu_b 0 \
    --sigma_a -1 \
    --sigma_b 3 \
    --epochs 20 \
    --hyperits 10 \
    --task reg > public_job_$i.out &
  sleep 10
done
wait
