# 生成某一帧的图片
python ./main/run.py --input_pose ./data/P01_01_00_0.pkl --begin_frame 30
# 生成一段视频
python ./main/run.py --input_pose ./data/P01_01_00_0.pkl --begin_frame 30 -- end_frame 50 --fps 20
# 生成整个视频
python ./main/run.py --input_pose ./data/P01_01_00_0.pkl --begin_frame 0 -- end_frame 1000 --fps 20