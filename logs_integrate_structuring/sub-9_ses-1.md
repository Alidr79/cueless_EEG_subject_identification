# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-09/ses-01/behavioral/sub-09_ses-01_commands.csv
Input EEG .tdsm file: Device Output Data/sub-09/ses-01/eeg/sub-09_ses-01.tdms
keyboard responses value counts:
 word_keyboard_response.keys
forward     21
stop        19
left        19
backward    18
right       18
missed       5
Name: count, dtype: int64
Removing extra triggers
index 10 remove
index 46 remove
Valid ✅
Number of valid commands : 95
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-09/ses-01/eeg/sub-09_ses-01_eeg.fif
