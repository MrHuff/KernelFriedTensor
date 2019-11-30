for i in {1..5}
do
  nohup taskset -c 0,1 python run_linear_benchmarks.py --seed $i --y_name 'rating' --data_path movielens_parquet_hashed_scaled --SAVE_PATH ./movielens_linear_benchmarks > linear_movielens_job_$i.out &
  sleep 10
done

