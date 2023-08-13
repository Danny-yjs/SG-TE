SG-TE
=======

# Requirements

* python = 3.8.13
* pytorch = 1.7.0
* matplotlib = 3.5.3
* sklearn = 1.1.3
* numpy = 1.23.3
* einops = 0.6.0
* timm = 0.6.11

# Training

* Step 1: download BRED dataset, and make sure it have the structure like following:
  
```python
./BRED/
   game/
      subject_0/
         Anger/
            openpose_output/json/
               0001_keypoints.json
               ...
               0075_keypoints.json
            cnn_features
	 ...
	 Happiness/
            ...
      ...
      subject_27/
   pre_game/
      subject_1/
      ...
      subject_28/
   annotations.csv

[Note] 0: Happiness; 1: Sadness; 2: Surprise; 3: Fear; 4: Disgust; 5: Anger 
```


* Step 2: create a folder called weights to hold the model files

* Step 3: change _annotations.csv_path_ in `datasets.py` to your path

* Step 4: get the data path and labels from the `annotations.csv` file in your BRED dataset path; get the coordinates of face, bodies, hands_right, hands_left of each frame from the json file; then assign them to data in `main.py`

* Step 5: run `python main.py`
