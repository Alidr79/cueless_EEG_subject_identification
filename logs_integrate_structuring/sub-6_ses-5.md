# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-06/ses-05/behavioral/sub-06_ses-05_commands.csv
Input EEG .tdsm file: Device Output Data/sub-06/ses-05/eeg/sub-06_ses-05.tdms
keyboard responses value counts:
 word_keyboard_response.keys
forward     12
right       11
stop        10
backward     8
left         7
missed       2
Name: count, dtype: int64
Removing extra triggers
index 13 remove
index 56 remove
Valid ✅
Number of valid commands : 48
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-06/ses-05/eeg/sub-06_ses-05_eeg.fif
