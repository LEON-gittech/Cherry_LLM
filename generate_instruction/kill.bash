for pid in $(ps -eo pid | grep -E '^ *9[0-9]{4}$'); do
  kill -9 $pid
done