detection.py: use FasterRCNN
run.py: use openpose
	python run.py --model=mobilenet_thin --resize=432x368 --video=./video/cam4.avi
	run.py is from tf_openpose
	so i just copy from that, to record my code
combine_detection.py: use detection.py's result to combine detection
	so camera 1~4's detection will become just one detection(after camera calibration)
	