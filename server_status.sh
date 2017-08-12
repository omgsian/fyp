
if ps -ef | grep -v grep | grep StreamingModule.py ; then
        exit 0
else
        python ./StreamingModule.py &
        #Write note to Logfile
        echo "[`date`]: twitter.py was not running... Restarted" >> ./log/harvest.log
        exit 0
fi