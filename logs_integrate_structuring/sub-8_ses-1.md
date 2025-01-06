# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-08/ses-01/behavioral/sub-08_ses-01_commands.csv
Input EEG .tdsm file: Device Output Data/sub-08/ses-01/eeg/sub-08_ses-01.tdms
keyboard responses value counts:
 word_keyboard_response.keys
stop        22
backward    19
right       19
left        18
forward     18
missed       4
Name: count, dtype: int64
Removing extra triggers
index 64 remove
index 71 remove
index 179 remove
Valid ✅
Number of valid commands : 96
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-08/ses-01/eeg/sub-08_ses-01_eeg.fif
