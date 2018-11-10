# Converted Voice Detection

"With Great Power Comes Great Responsibility" - Uncle Ben (Spiderman 2002). 

Voice Conversion is the transformation of one speaker’s voice (the source) to sound like another speakers voice (the target). 
As voice conversion tools improve, generated and authentic voice clips become increasingly indiscernible to the human ear which makes the dissemination of spurious information much more conducive.  If you are interested in learning an effective approach to converted voice detection, you have come to the right place. For a more detailed account of this project, check out the [paper](https://github.com/ciads-ut/converted-voice-detection/blob/master/Paper%20%26%20Presentation/paper.pdf)
and [presentation](https://github.com/ciads-ut/converted-voice-detection/blob/master/Paper%20%26%20Presentation/Voice_Conversion_Presentation.pdf).

## Voice Conversion
The voice conversion tool used in this demo is called Sprocket. Sprocket is an open source voice conversion tool that allows for users to convert one speaker’s voice (the source) to another speakers voice (the target). The creator of Sprocket, Kazuhiro Kobayashi, wanted to develop a tool that allowed users to easily test voice conversion from the comfort of their own computer. To learn more about Sprocket, visit their [website](https://github.com/k2kobayashi/sprocket).

## Getting Started
#### Requirements:
Please use Python3.
```sh
pip install -r requirements.txt
```
#### Data Exploration
For this demo, we are using data from the Voice Conversion Challenge of 2016. To see what the dataset looks like, run the following command.
```sh
cd Example
```
![folders](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/folders.PNG)

There were a total of 10 speakers and thus, 10 folders holding the recordings for the respective speakers.
  - Each folder contains the same 216 utterances from the book “The Call of the Wild” by Jack London
  - Folders that start with "S" are source folders
  - Folders that start with "T" are target folders
  - Folder names that contain "F" represents a female speaker
  - Folder names that contain "M" represents a male speaker

I decided to do two of every gender to gender combination. In other words, there were two source male to target male conversions, two source male to target female conversion, etc. In total, there were 8 new folders that contain converted voices. (216 WAV files each)
- Converted Folders end with a ratio that represents how similar the source and target speakers were
- Ratios close to 1 imply that the source and target spoke in similar frequencies

In total, we have 10 folders that contain authentic voice clips and 8 folders that contain converted voice clips which yields 2160 authentic WAV files and 1728 generated WAV files.
## Analysis
#### Signal Visualization
We will begin with analyzing two signals: one real and one converted. The commands below will show you how this can be done. Do you see any difference between the signals?
```sh
cd Visualizations/Wave
python wav.py
```

![signals](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/signals.png)


From this analysis, it seemed quite apparent that the lows for each of the signals differed ever so slightly. When it came to lows in the signals, authentic voice clips (both target and source) had a greater range of wave amplitudes than that of generated voice clips. This trend seemed recur throughout the corpra.

#### MFCC
To better assess this discrepancy, I decided to extract the MFCC from each of the WAV files. MFCC analyzes amplitudes of sound by decomposing the signal to several, smaller frequencies. These frequencies are mapped onto a Mel-Scale Filter Bank that gives us a numerical representation of the original sound. For more information about MFCC, click [here](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). Run the following commands to extract the MFCC from all 18 folders (2160 authentic WAV files and 1728 generated WAV files) by using the python library speech_features. 

```sh
cd Example
python mfcc_test.py
```

These 52 features go along with a class label of real or fake, based on which folder the WAV file came from. In total, each WAV file has 53 features: 52 MFCC features and 1 class label (real or fake). All 53 features for each of the 3888 WAV files are written into a CSV that will be used to train a machine learning model.The picture below shows a screenshot of the first 14 features for a few WAV files. 

![csv](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/csv.PNG)


#### T-SNE
With a total of 53 features, visualizing our data can be tricky. This is where we can use a dimensional reduction approach called t-SNE to get an idea what our data looks like. Using t-SNE open-source code by Siraj Raval, I was able to get a sense of the data by reducing it to two dimensions. From this visualization, it was evident that there was some congregation of converted voices and authentic voices based on the given MFCC values. Run the command below to see for yourself!

```sh
cd Visualizations/T-SNE/
python data_visualization.py
```
![tsne](https://user-images.githubusercontent.com/25602219/44238346-f3e84380-a179-11e8-84db-ddcabdb323f9.png)

## Machine Learning

#### Logistic Regression
Our data has 53 features: 52 MFCC features and one class label (real or fake). We are curious to see how well the 52 MFCC features are in predicting the class label of real or fake. Due to the binary state of the dependent variable (class label), I decided to use logistic regression. The platform used to conduct the Logistic Regression was in R. Since we had 1728 converted voices and 2160 authentic voices, I decided to keep this ratio of converted to authentic voices the same in both training and test data sets. I used 80% of the 3888 files as training data and the remaining 20% as test data. The RMD to analyze our csv is called mfcc.rmd and can be found in this directory:

```sh
cd Example
```

If you do not have R installed, click [here](http://rpubs.com/v4lakers/mfcc) to view the R markdown online.

#### Results
Once the model was created using the 53 features, I did some evaluation on the model performance. The first model analysis I undertook was a density plot that seemed to separate fake and converted voices well on the training data. There seems to be a pretty good separation between the two classes. 

![density](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/density.png)


The next form of analysis were plotting the ROC curves for both the training data and test data. The results showed that both training and test data had about .98 area under the curve.

![roc](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/roc.png)


The final form of analysis was assessing the model’s performance on just the test data. The model had an accuracy of 81.3\% while precision was 99.5\% and recall was 75\%. These values indicate that the MFCC was indeed a powerful tool in differentiating between converted voices and authentic voices.


![acc](https://github.com/ciads-ut/converted-voice-detection/blob/master/Visualizations/acc.png)


#### Conclusion
The signals of real voice clips and converted voice clips differed in the lows as real voice clips seemed to have slightly higher amplitudes than converted voice clips, on average. Using MFCC, we were able to assess this discrepancy by decomposing each signal and representing it as a matrix of numbers. While this form of analysis gave us good results in differentiating between real and converted voice clips, there are a few things that need to be addressed. With a precision of 99\%, it seems that the model did extremely well in predicting clips that were authentic. On the other hand, with a recall of 75\%, our model could have done better in predicting converted voice clips. Future expiriments will address other features to go along with MFCC to aid in detection.
