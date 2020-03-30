for i in {1..5}
do
  taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.05 \
    --max_lr 1e-1 \
    --max_R 100 \
    --PATH ./report_movielens_data_ml-10m/ \
    --reg_para_a 50 \
    --reg_para_b 500 \
    --save_path report_movielens_univariate_10m_split_2 \
    --seed $i \
    --split_mode 2 \
    --tensor_name all_data.pt \
    --architecture 2 \
    --side_info_order 1 \
    --latent_scale False \
    --dual True \
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
