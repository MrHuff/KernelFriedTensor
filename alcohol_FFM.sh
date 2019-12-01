for i in {1..5}
do
  nohup taskset -c $i python run_FFM_benchmarks.py --seed $i --y_name 'Bottles Sold' --data_path alcohol_sales_parquet_FFM --SAVE_PATH ./alcohol_benchmarks > FFM_alcohol_job_$i.out &
  sleep 10
done

