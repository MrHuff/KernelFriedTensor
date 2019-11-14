nohup taskset -c 0,1 python run_job_script.py --max_lr 1e-1 --max_R 100 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed 1 --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt > public_job_1.out &
sleep 10
nohup taskset -c 2,3 python run_job_script.py --max_lr 1e-1 --max_R 100 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed 2 --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt > public_job_2.out &
sleep 10
nohup taskset -c 4,5 python run_job_script.py --max_lr 1e-1 --max_R 100 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed 3 --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt > public_job_3.out &
sleep 10
nohup taskset -c 6,7 python run_job_script.py --max_lr 1e-1 --max_R 100 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed 4 --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt > public_job_4.out &
sleep 10
nohup taskset -c 8,9 python run_job_script.py --max_lr 1e-1 --max_R 100 --PATH ./public_data/ --reg_para_a 1e-2 --reg_para_b 100. --fp_16 True --fused True --save_path public_job --seed 5 --tensor_name all_data.pt --architecture 1 --side_info_order 1 0 2 --temporal_tag 2 --task reg --tensor_name public_data_tensor.pt --side_info_name public_article_tensor.pt public_location_tensor.pt public_time_tensor.pt > public_job_5.out &
sleep 10

