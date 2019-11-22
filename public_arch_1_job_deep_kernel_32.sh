for i in {1..5}
do
  nohup taskset -c 0,1 python run_job_script.py --deep_kernel True --max_lr 2e-2 --max_R 100 --PATH ./public_data/ --reg_para_a 0 --reg_para_b 100. --fp_16 False --fused True --save_path public_job_deep_arch_1 --seed $i --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg > public_job_$i.out &
  sleep 10
done
