for i in {1..5}
do
  taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.05 \
    --max_lr 1e-1 \
    --max_R 6 \
    --PATH ./public_movielens_data/ \
    --reg_para_a 3 \
    --reg_para_b 10 \
    --save_path movielens_job_arch_0_benchmark_bayesian_univariate \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 2 \
    --delete_side_info 1 2 \
    --old_setup True \
    --latent_scale False \
    --dual False \
    --multivariate False \
    --init_max 1e-1 \
    --bayesian True \
    --mu_a 0 \
    --mu_b 0 \
    --sigma_a -2 \
    --sigma_b 0 \
    --epochs 20 \
    --hyperits 10 \
    --task reg > movielens_job_$i.out &
  sleep 10
done
wait
