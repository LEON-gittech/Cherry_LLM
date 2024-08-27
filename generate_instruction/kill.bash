for pid in $(ps -eo pid | grep -E '^ *19[0-9]{5}$'); do
  kill -9 $pid
done