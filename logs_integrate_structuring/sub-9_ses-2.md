# Results

Input BEHAVIORAL .csv file: Device Output Data/sub-09/ses-02/behavioral/sub-09_ses-02_commands.csv
Input EEG .tdsm file: Device Output Data/sub-09/ses-02/eeg/sub-09_ses-02.tdms
keyboard responses value counts:
 word_keyboard_response.keys
left        22
stop        22
right       20
forward     19
backward    17
Name: count, dtype: int64
Removing extra triggers
index 5 remove
index 36 remove
index 45 remove
index 111 remove
Valid ✅
Number of valid commands : 100
Columns after rename:
 Index(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4',
       'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
       'P7', 'P3', 'Pz', 'P4', 'P8', 'O2', 'Oz', 'O1', 'Event'],
      dtype='object')
Number of columns: 31
🟢Exported EDF file saved to: Imagined Command Dataset/sub-09/ses-02/eeg/sub-09_ses-02_eeg.fif
