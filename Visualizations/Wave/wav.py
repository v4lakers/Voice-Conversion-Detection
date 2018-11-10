# Import Necessary Libraries
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Define a Signal Reading Function
def signal(file):

    # Open Wav Files
    y= wave.open(file, 'r')

    # Retrieve and Plot Signal Information
    samplerate, data = wavfile.read(file)
    plt.plot(data)

    # Add Axis-Labels and a Title
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal For "+str(file))

    # Display Signal Visualization
    plt.show()

# Pass Sample Wav Files for Visualization
signal('sample_real.wav')
signal('sample_fake.wav')
