# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-11/ses-03/behavioral/sub-11_ses-03_commands.csv
Input EEG .tdsm file: Device Output Data/sub-11/ses-03/eeg/sub-11_ses-03.tdms
keyboard responses value counts:
 word_keyboard_response.keys
stop        23
right       19
left        18
backward    18
forward     16
missed       6
Name: count, dtype: int64
Removing extra triggers
index 17 remove
index 96 remove
Valid ✅
Number of valid commands : 94
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-11/ses-03/eeg/sub-11_ses-03_eeg.fif
