# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-05/ses-02/behavioral/sub-05_ses-02_commands.csv
Input EEG .tdsm file: Device Output Data/sub-05/ses-02/eeg/sub-05_ses-02.tdms
keyboard responses value counts:
 word_keyboard_response.keys
right       24
backward    23
left        19
forward     17
stop        15
missed       2
Name: count, dtype: int64
Removing extra triggers
index 62 remove
index 156 remove
index 194 remove
Valid ✅
Number of valid commands : 98
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-05/ses-02/eeg/sub-05_ses-02_eeg.fif
