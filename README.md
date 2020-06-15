# eeg seizure detection
 
 <h2>Overview</h2>
Created for [Neureka](https://neureka-challenge.com), a 2020 competition for seizure detection in scalp electrocencephalogram (EEG) readings. The dataset used in the competition was taken from the [Temple University EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/), and submitted under the team name "RocketShoes". Our general approach was to use a pretrained ResNet18 model on overlapping windows of Short-Time Fourier Transformed (STFT) and preprocessed voltage readings to predict on the binary classification task of seizure versus background classes. 

<h2>Preprocessing</h2>
