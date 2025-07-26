ps -ef | grep evaluate_a_set_of_videos.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep evaluate_senorita.sh | grep -v grep | awk '{print $2}' | xargs kill -9