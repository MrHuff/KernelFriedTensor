for i in {1..5}
do
  nohup taskset -c $i python run_job_script.py \
    --batch_size_a 0.01 \
    --batch_size_b 0.10 \
    --max_lr 1e-1 \
    --max_R 35 \
    --PATH ./public_data/ \
    --reg_para_a 0 \
    --reg_para_b 1 \
    --save_path public_job_arch_0_primal_old \
    --seed $i \
    --tensor_name all_data.pt \
    --architecture 0 \
    --side_info_order 1 0 2 \
    --temporal_tag 2 \
    --latent_scale False \
    --dual False \
    --init_max 0.1 \
    --old_setup True \
    --task reg > public_job_$i.out &
  sleep 10
done

