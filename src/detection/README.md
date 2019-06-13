detection.py: use FasterRCNN
run.py: use openpose
	dir = C:\Users\Owner\Anaconda3\envs\tensorflow\Lib\site-packages\tf-openpose
	python run.py --model=mobilenet_thin --resize=432x368 --video=./video/cam2.avi
	run.py is from tf_openpose
	so i just copy from that, to record my code
combine_detection.py: use detection.py's result to combine detection
	so camera 1~4's detection will become just one detection(after camera calibration)
	


use AphlaPose:
	cd D:\Code\AlphaPose-pytorch
	python video_demo.py --video cam1.avi --outdir examples\res --save_video --sp

use OpenPose:
	cd D:\Code\tf-pose-estimation-master
	python run_video.py --model=cmu --resolution=432x368 --video=D:/Code/MultiCamOverlap/dataset/videos/Player01/track8/cam1.avi