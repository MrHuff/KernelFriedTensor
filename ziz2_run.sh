for i in {0..0}
do
  python run_job_script_preloaded.py --idx=$i --job_path="jobs_traffic_baysian_WLR_3" > ziz2_job_$i.out &
  sleep 10
done
wait

#for i in {0..2}
#do
#  python run_job_script_preloaded.py --idx=$i --job_path="jobs_CCDS_baysian_LS_3" > ziz2_job_$i.out &
#  sleep 10
#done
#wait