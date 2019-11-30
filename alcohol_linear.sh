for i in {1..5}
do
  nohup taskset -c 0,1 python run_linear_benchmarks.py --seed $i --y_name 'Bottles Sold' --data_path alcohol_sales_parquet_hashed_scaled --SAVE_PATH ./alcohol_linear_benchmarks > linear_alcohol_job_$i.out &
  sleep 10
done

