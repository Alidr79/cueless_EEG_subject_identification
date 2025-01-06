# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-06/ses-01/behavioral/sub-06_ses-01_commands.csv
Input EEG .tdsm file: Device Output Data/sub-06/ses-01/eeg/sub-06_ses-01.tdms
keyboard responses value counts:
 word_keyboard_response.keys
right       33
left        18
forward     17
backward    16
stop        10
missed       6
Name: count, dtype: int64
Removing extra triggers
index 47 remove
index 121 remove
index 128 remove
index 142 remove
Valid ✅
Number of valid commands : 94
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-06/ses-01/eeg/sub-06_ses-01_eeg.fif
