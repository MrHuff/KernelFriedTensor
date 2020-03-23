for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.1 \
    --max_lr 1e-1 \
    --max_R 1000 \
    --PATH ./report_movielens_data_ml-1m/ \
    --reg_para_a 0 \
    --reg_para_b 1e-1 \
    --save_path report_movielens_1m_job_arch_2_primal \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 2 \
    --side_info_order 1 0 \
    --latent_scale False \
    --dual True \
    --init_max 1e-1 \
    --task reg > report_movielens_job_$i.out &
  sleep 10
done
