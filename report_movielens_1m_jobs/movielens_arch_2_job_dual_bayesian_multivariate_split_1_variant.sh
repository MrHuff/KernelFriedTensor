for i in {1..5}
do
  taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.05 \
    --full_grad True \
    --max_lr 1e-1 \
    --max_R 2000 \
    --PATH ./report_movielens_data_ml-1m/ \
    --reg_para_a 1000 \
    --reg_para_b 10000 \
    --save_path report_movielens_multivariate_1m_split_1_variant \
    --seed $i \
    --split_mode 1 \
    --tensor_name all_data.pt \
    --architecture 2 \
    --side_info_order 1 0 \
    --delete_side_info 0 \
    --latent_scale False \
    --dual True \
    --multivariate True \
    --init_max 1e-1 \
    --bayesian True \
    --mu_a 0 \
    --mu_b 0 \
    --sigma_a -1 \
    --sigma_b 3 \
    --task reg > movielens_job_$i.out &
  sleep 10
done
wait