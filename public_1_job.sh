for i in 1
  do
    nohup python run_job_script.py --max_R 30 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 1e2 --fp_16 True --fused True --save_path public_job --seed $i --tensor_name all_data.pt --architecture 0 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt &
    sleep 10
  done

