for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.15 \
    --max_lr 1e-1 \
    --max_R 10 \
    --PATH ./public_data/ \
    --reg_para_a 0 \
    --reg_para_b 5 \
    --fp_16 False \
    --fused True \
    --save_path movielens_job_arch_0 \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual False \
    --task reg > movielens_job_$i.out &
  sleep 10
done

