for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.1 \
    --full_grad True \
    --max_lr 1e0 \
    --max_R 500 \
    --PATH ./report_movielens_data_ml-10m/ \
    --reg_para_a 0 \
    --reg_para_b 1e-7 \
    --save_path report_movielens_10m_job_arch_2_dual_split_2 \
    --seed $i \
    --split_mode 2 \
    --tensor_name all_data.pt \
    --architecture 2 \
    --side_info_order 1 \
    --latent_scale False \
    --dual True \
    --init_max 1e-1 \
    --task reg > report_movielens_job_$i.out &
  sleep 10
done
wait