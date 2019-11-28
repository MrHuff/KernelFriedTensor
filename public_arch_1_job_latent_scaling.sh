for i in {1..5}
do
  nohup taskset -c 0,1 python run_job_script.py --batch_size_a 0.01 --batch_size_b 0.1 --max_lr 1e0 --max_R 100 --PATH ./public_data/ --reg_para_a 0 --reg_para_b 100. --fp_16 False --fused True --save_path public_job_arch_1_latent_scale --seed $i --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --latent_scale True > public_job_$i.out &
  sleep 10
done

