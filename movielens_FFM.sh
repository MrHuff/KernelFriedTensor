for i in {1..5}
do
  nohup taskset -c $i python run_FFM_benchmarks.py --seed $i --y_name 'rating' --data_path movielens_parquet_FFM --SAVE_PATH ./movielens_benchmarks > FFM_movielens_job_$i.out &
  sleep 10
done

