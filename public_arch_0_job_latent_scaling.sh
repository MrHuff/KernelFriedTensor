for i in {1..5}
do
  nohup taskset -c 0,1 python run_job_script.py --batch_size_a 0.01 --batch_size_b 0.1 --max_lr 1e-1 --max_R 20 --PATH ./public_data/ --reg_para_a 0 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed $i --tensor_name all_data.pt --architecture 0 --side_info_order 1 0 2 --temporal_tag 2 --task reg --latent_scale True> public_job_$i.out &
  sleep 10
done

