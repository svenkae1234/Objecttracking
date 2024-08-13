# Objecttracking
## Watch Videos from Objecttracking
Download Videos from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpofcWmifuVek-O4tA?e=OcwXdz
Select the Video of the Model you want to watch

## Run without docker
#### Setup
##### Install Requirements
```
pip install tensorflow==2.15.1
pip install tensorflow[and-cuda]
pip install torch==2.3.1
pip install torchvision==0.18.1
pip install pandas
pip install ultralytics
pip install matplotlib
pip install opencv-python
```
##### Download dataset
Download dataset from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoUWfBcFrRoqDCeLw?e=SMNNP6
Unzip dataset in to ./Objecttracking/src folder

### Run CNN-Model
##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoXH8_dkSf_26dz5Q?e=0ZobZh
Copy Model to src/best_models/CNN/model
```
cd Objecttracing/src/best_models/CNN/evaluation
python3 prediction_time.py
```

### Run Faster-RCNN Model
##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoWD5w9wh_KUKBHfQ?e=oOGUDq
Copy Model to src/best_models/faster_rcnn_model/model
```
cd Objecttracing/src/best_models/faster_rcnn_model/evaluation
python3 prediction_time.py
```

### Run Yolo-Model
##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoVUfnEomFX6aohRg?e=fRenRW
Copy Model to Objecttracking/src/best_models/yolo_model/model
```
cd Objecttracing/src/best_models/yolo_model/evaluation
python3 prediction_time.py
```

## Run with docker
Make sure Docker is installed
#### Setup
##### Clone this repo
```
git clone https://github.com/svenkae1234/Objecttracking.git
```
##### Download dataset
Download dataset from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoUWfBcFrRoqDCeLw?e=SMNNP6
Unzip dataset in to ./Objecttracking/src folder

##### Start docker container
```
cd Objecttracking
docker-compose build
docker-compose up
```

### Run CNN-Model

##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoXH8_dkSf_26dz5Q?e=0ZobZh
Make sure you setup the Docker Container
Switch to the Docker Container
Copy Model to src/best_models/CNN/model
```
cd src/best_models/CNN/evaluation
python3 prediction_time.py
```
Open: http://localhost:8080/vnc.html?autoconnect=1&resize=scale

### Run Faster-RCNN Model
##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoWD5w9wh_KUKBHfQ?e=oOGUDq
Make sure you setup the Docker Container
Switch to the Docker Container
Copy Model to src/best_models/faster_rcnn_model/model
```
cd src/best_models/faster_rcnn_model/evaluation
python3 prediction_time.py
```
Open: http://localhost:8080/vnc.html?autoconnect=1&resize=scale

### Run Yolo-Model
##### Download Model
Download Model from: https://1drv.ms/u/s!Ar4VGyiVG1e0zpoVUfnEomFX6aohRg?e=fRenRW
Make sure you setup the Docker Container
Switch to the Docker Container
Copy Model to src/best_models/yolo_model/model
```
cd src/best_models/yolo_model/evaluation
python3 prediction_time.py
```
Open: http://localhost:8080/vnc.html?autoconnect=1&resize=scale

