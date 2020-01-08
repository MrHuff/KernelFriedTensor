for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.10 \
    --max_lr 1e-1 \
    --max_R 14 \
    --PATH ./public_movielens_data/ \
    --reg_para_a 0 \
    --reg_para_b 2 \
    --fp_16 False \
    --fused True \
    --save_path movielens_job_arch_0_primal_old \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --old_setup True \
    --dual False \
    --init_max 1e-1 \
    --task reg > movielens_job_$i.out &
  sleep 10
done

