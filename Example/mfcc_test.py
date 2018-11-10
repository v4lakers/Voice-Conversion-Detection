# Import Necessary Libraries
import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

# Function that takes in Sound & Outputs MFCC Matrix
# MFCC Extraction Code from https://github.com/jameslyons/python_speech_features
# James Lyons
def extract(sound):
    (rate,sig) = wav.read(sound)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    matrix = (fbank_feat[1:3,:])
    return matrix

# Create and Open a CSV to Store MFCC Values
with open("mfcc.csv", "w") as file:

    # Initialize First Line as Column Names
    line = "Type"
    for i in range(0,52):
        line = line+", f"+str(i)
    line = line+", \n"
    file.write(line)

# Append to the CSV with MFCC values for each
with open("mfcc.csv","a") as file:

    #  Get Current Working Directory
    owd = os.getcwd()

    # Loop through each Folder
    for folder in os.listdir():

        # Change Working Directory to Folder
        if 'mfcc' not in folder and folder != '.Rhistory':
            os.chdir(folder)

            # Loop through each WAV File
            for sound in os.listdir():

                    # MFCC Function
                    matrix = extract(sound)

                    # Initialize First Cell as Class Label
                    if len(folder) == 3:
                        line = "real"
                    else:
                        line = "fake"

                    # Write each MFCC value of Matrix to the CSV
                    for column in matrix:
                        for item in column:
                            line = line+", "+str(item)
                    line = line+", \n"
                    file.write(line)

        # Return to Previous Directory
        os.chdir(owd)
