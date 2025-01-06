# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-03/ses-05/behavioral/sub-03_ses-05_commands.csv
Input EEG .tdsm file: Device Output Data/sub-03/ses-05/eeg/sub-03_ses-05.tdms
keyboard responses value counts:
 word_keyboard_response.keys
left        14
right       13
forward      9
backward     7
stop         6
Name: count, dtype: int64
Removing extra triggers
index 3 remove
index 6 remove
index 25 remove
index 40 remove
index 82 remove
Valid ✅
Number of valid commands : 49
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-03/ses-05/eeg/sub-03_ses-05_eeg.fif
