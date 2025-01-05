import os
import pandas as pd
from tqdm import tqdm
from nptdms import TdmsFile
import numpy as np
import mne
import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert device output data to .fif format with annotations')
    parser.add_argument('--subject-num', type=int, required=True, help='Subject id (integer) starts from 1')
    parser.add_argument('--session-num', type=int, required=True, help='Session id (integer) start from 1')
    args = parser.parse_args()

    SUBJECT_NUM = args.subject_num
    SESSION_NUM = args.session_num

    # Define directories and paths
    BEHAVIORAL_DIR = get_behavioral_dir(SUBJECT_NUM, SESSION_NUM)
    BEHAVIORAL_PATH = get_behavioral_file(BEHAVIORAL_DIR)
    EEG_RAW_PATH = get_eeg_raw_path(SUBJECT_NUM, SESSION_NUM)

    print("BEHAVIORAL .csv file:", BEHAVIORAL_PATH)
    print("EEG .tdsm file:", EEG_RAW_PATH)


    LOGS_DIR = 'logs_integrate_structuring'
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_results = []
    log_results.append(f"Input BEHAVIORAL .csv file: {BEHAVIORAL_PATH}")
    log_results.append(f"Input EEG .tdsm file: {EEG_RAW_PATH}")


    # Read and process behavioral data
    behavioral_data = pd.read_csv(BEHAVIORAL_PATH, sep='\t')
    word_labels = process_behavioral_data(behavioral_data, log_results)

    # Read EEG data from TDMS file
    df_eeg_raw = read_eeg_data(EEG_RAW_PATH)

    # Process EEG triggers
    start_stop_indices = process_eeg_triggers(df_eeg_raw)

    # Remove extra triggers
    remove_extra_triggers(start_stop_indices, log_results)

    # Validate triggers
    trigger_validity(start_stop_indices, log_results)
    if SESSION_NUM <= 3:
        assert len(start_stop_indices) == 200, "Error : still wrong number of start_stop_indices!"

    if SESSION_NUM > 3:
        if (SUBJECT_NUM != 3 or SESSION_NUM != 5):# Exception for the: sub-03_ses-05
            if (SUBJECT_NUM != 10 or SESSION_NUM != 4):# Exception for the: sub-10_ses-04
                assert len(start_stop_indices) == 100, "Error : still wrong number of start_stop_indices!"

    # Create 'Event' column in EEG DataFrame
    trigger_starts = [start_stop_indices[i] for i in range(len(start_stop_indices)) if i % 2 == 0]
    df_eeg_raw = create_event_column(df_eeg_raw, trigger_starts, word_labels, log_results)

    # Clean EEG DataFrame
    df_eeg_raw = clean_eeg_dataframe(df_eeg_raw, log_results)

    # Create MNE Raw object
    mne_raw_eeg = create_mne_raw(df_eeg_raw)


    if SUBJECT_NUM == 10 and SESSION_NUM == 4:
        # Crop exception for this session
        # i forget to put the loop in psychopy to 50, and subject hit some more trials 
        # so after all event labeling and removing missed trials, i cropped the signal correctly for 50 first healthy trials.
        events = mne.find_events(mne_raw_eeg)
        fiftieth_event_sec = events[49][0] / mne_raw_eeg.info['sfreq']
        mne_raw_eeg.crop(tmax = fiftieth_event_sec + 5) # with 5 second after it.(i check that it doesn't overlap with the next event that i removed.)

    # Add annotations
    mne_raw_eeg = add_annotations(mne_raw_eeg)

    # Drop 'Event' channel before exporting
    mne_raw_eeg.drop_channels(['Event'])

    print("-------------")
    print(mne_raw_eeg.get_data().min())
    print(mne_raw_eeg.get_data().max())
    print("-------------")
    
    # Export to fif
    export_to_fif(mne_raw_eeg, SUBJECT_NUM, SESSION_NUM, log_results)

    save_results_to_markdown(log_results, f"{LOGS_DIR}/sub-{SUBJECT_NUM}_ses-{SESSION_NUM}.md")


def get_behavioral_dir(subject_num, session_num):
    """Construct the behavioral directory path based on subject and session ids."""
    subject_str = f'sub-{subject_num:02d}'
    session_str = f'ses-{session_num:02d}'
    behavioral_dir = os.path.join('Device Output Data', subject_str, session_str, 'behavioral')
    return behavioral_dir


def get_behavioral_file(behavioral_dir):
    """Find the behavioral file in the specified directory."""
    files = [f for f in os.listdir(behavioral_dir) if os.path.isfile(os.path.join(behavioral_dir, f))]
    if len(files) != 1:
        raise Exception(f"Error: Expected exactly one behavioral file in {behavioral_dir}, found {len(files)}.")
    behavioral_file = os.path.join(behavioral_dir, files[0])
    return behavioral_file


def get_eeg_raw_path(subject_num, session_num):
    """Construct the EEG TDMS file path based on subject and session ids."""
    subject_str = f'sub-{subject_num:02d}'
    session_str = f'ses-{session_num:02d}'
    eeg_raw_file = f'{subject_str}_{session_str}.tdms'
    eeg_raw_path = os.path.join('Device Output Data', subject_str, session_str, 'eeg', eeg_raw_file)
    return eeg_raw_path


def process_behavioral_data(behavioral_data, log_results):
    """Process the behavioral data and map keyboard responses to labels."""
    word_labels = behavioral_data[['word_keyboard_response.keys']].copy()
    replacements = {
        '0': 'stop',
        'up': 'forward',
        'down': 'backward',
        'return': 'missed'
    }
    word_labels['word_keyboard_response.keys'].replace(replacements, inplace=True)
    print(word_labels['word_keyboard_response.keys'].value_counts())
    log_results.append(f"keyboard responses value counts:\n {word_labels['word_keyboard_response.keys'].value_counts()}")

    return word_labels


def read_eeg_data(eeg_raw_path):
    """Read EEG data from the TDMS file and correct channel names."""
    eeg_raw = TdmsFile.read(eeg_raw_path)
    df_eeg_raw = eeg_raw.as_dataframe(time_index=False)

    # Correct the column names
    df_eeg_raw.rename(columns=lambda x: x.replace("/'EEG_Raw'/", ''), inplace=True)
    df_eeg_raw.rename(columns=lambda x: x.replace("'", ''), inplace=True)

    return df_eeg_raw


def process_eeg_triggers(df_eeg_raw):
    """Process EEG triggers to identify start and stop indices."""
    eeg_trigger = df_eeg_raw[['TRIG']].rename(columns={'TRIG': 'Trigger'})
    temp_df = (eeg_trigger - \
        eeg_trigger.shift(1))
    
    start_stop_indices = temp_df[temp_df['Trigger'] > +1].index.values.tolist()
    return start_stop_indices


def remove_extra_triggers(input_start_stop_indices, log_results):
    """Remove extra triggers that do not correspond to valid events. The origin of this triggers is unknown for us!"""
    log_results.append("Removing extra triggers")

    for i in range(0,len(input_start_stop_indices)):
        if i == 0: # for first
            diff_next = input_start_stop_indices[i+1] - input_start_stop_indices[i]
            if not(495 <= diff_next <= 505) :
                print(f"index {i} remove")
                log_results.append(f"index {i} remove")
                input_start_stop_indices.remove(input_start_stop_indices[i])

        elif i == len(input_start_stop_indices)-1 : # for last
            diff_prev = input_start_stop_indices[i] - input_start_stop_indices[i-1]
            if not(495 <= diff_prev <= 505) :
                print(f"index {i} remove")
                log_results.append(f"index {i} remove")
                input_start_stop_indices.remove(input_start_stop_indices[i])

        elif 0 < i < len(input_start_stop_indices)-1 : # for other elements
            diff_prev = input_start_stop_indices[i] - input_start_stop_indices[i-1]
            diff_next = input_start_stop_indices[i+1] - input_start_stop_indices[i]

            if i == len(input_start_stop_indices) - 2 : # before last one
                if not(495 <= diff_prev <= 505) and not(495 <= diff_next <= 505) :
                    print(f"index {i} remove!")
                    log_results.append(f"index {i} remove")
                    input_start_stop_indices.remove(input_start_stop_indices[i])

            else : 
                diff_next2 = input_start_stop_indices[i+2] - input_start_stop_indices[i]
                
                if  not(495 <= diff_prev <= 505) and not(495 <= diff_next <= 505) and  not(495 <= diff_next2 <= 505) :
                    print(f"index {i} remove!")
                    log_results.append(f"index {i} remove")
                    input_start_stop_indices.remove(input_start_stop_indices[i])


def trigger_validity(input_start_stop_indices, log_results):
    for i in range(0,len(input_start_stop_indices),2):
        diff = input_start_stop_indices[i+1]-input_start_stop_indices[i] # sample difference between start and end

        if not(diff >= 495 and diff <= 505) :
            print("Difference =", diff)
            log_results.append(f"Difference = {diff}")
            log_results.append(f"❌Error at number of samples in epoch start = {input_start_stop_indices[i]}"
                            +f" , end = {input_start_stop_indices[i+1]}❌")
            
            raise Exception(f"❌Error at number of samples in epoch start = {input_start_stop_indices[i]}"
                            +f" , end = {input_start_stop_indices[i+1]}❌")
        
    print("Valid ✅")
    log_results.append("Valid ✅")



def create_event_column(df_eeg_raw, trigger_starts, word_labels, log_results):
    """Create the 'Event' column in the EEG DataFrame based on triggers and word labels."""
    word_labels_dict = {'backward': 1, 'forward': 2, 'left': 3, 'right': 4, 'stop': 5}
    df_eeg_raw['Event'] = 0
    behavioral_index = -1
    
    for i in tqdm(range(df_eeg_raw.shape[0])):
        if i in trigger_starts : 
            behavioral_index += 1 # behavioral_index shows the event corresponding word index in word_labels
            current_word = word_labels['word_keyboard_response.keys'].iloc[behavioral_index]

            if current_word != "missed": # Here we are removing missed trials
                df_eeg_raw.loc[i, 'Event'] = word_labels_dict[current_word]

    print("Number of valid commands :", len(df_eeg_raw[df_eeg_raw['Event'] != 0]))
    log_results.append(f"Number of valid commands : {len(df_eeg_raw[df_eeg_raw['Event'] != 0])}")
    return df_eeg_raw


def clean_eeg_dataframe(df_eeg_raw, log_results):
    """Remove unnecessary channels and rename EEG channels appropriately."""
    df_eeg_raw.drop(columns=['TRIG', 'KEYB', 'MOUSE'], inplace=True, errors='ignore')

    channel_renames = {'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'FCZ':'FCz',
                             'CPZ':'CPz', 'CPZ':'CPz', 'OZ':'Oz', 'CZ':'Cz', 'PZ':'Pz'}

    df_eeg_raw.rename(columns=channel_renames, inplace=True)

    print("Columns after rename:")
    print(df_eeg_raw.columns)
    print(f"Number of columns: {len(df_eeg_raw.columns)}")

    log_results.append(f"Columns after rename:\n {df_eeg_raw.columns}")
    log_results.append(f"Number of columns: {len(df_eeg_raw.columns)}")

    return df_eeg_raw


def create_mne_raw(df_eeg_raw):
    """Create an MNE Raw object from the EEG DataFrame."""
    ch_names = df_eeg_raw.columns.tolist()
    ch_types = ['eeg'] * (len(ch_names) - 1) + ['stim']  # Last channel is 'Event'
    sample_rate = 250  # Sampling frequency in Hz

    # Convert DataFrame to numpy array and transpose
    data = df_eeg_raw.values.T

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
    info.set_montage("standard_1020")

    # Create Raw object
    mne_raw_eeg = mne.io.RawArray(data, info)
    return mne_raw_eeg


def add_annotations(mne_raw_eeg):
    """Add annotations to the MNE Raw object based on events."""
    event_id = {1: 'backward', 2: 'forward', 3: 'left', 4:'right', 5:'stop'}

    events = mne.find_events(mne_raw_eeg, stim_channel='Event')
    # Convert events to annotations
    onset = events[:, 0] / mne_raw_eeg.info['sfreq']  # Convert sample indices to time in seconds
    description = [event_id[event] for event in events[:, 2]]

    annotations = mne.Annotations(onset=onset, duration = 2, description = description)
    mne_raw_eeg.set_annotations(annotations)

    return mne_raw_eeg


def export_to_fif(mne_raw_eeg, subject_num, session_num, log_results):
    """Export the MNE Raw object to a fif file."""
    subject_str = f'sub-{subject_num:02d}'
    session_str = f'ses-{session_num:02d}'
    output_dir = os.path.join('Dataset', subject_str, session_str, 'eeg')
    os.makedirs(output_dir, exist_ok=True)


    output_file = os.path.join(output_dir, f'{subject_str}_{session_str}_eeg.fif')
    mne_raw_eeg.save(output_file)

    print(f"🟢Exported fif file saved to: {output_file}")
    log_results.append(f"🟢Exported fif file saved to: {output_file}")


def save_results_to_markdown(results, filename):
    """Save results to a Markdown file."""
    with open(filename, 'w') as f:
        f.write("# Results\n\n")
        for result in results:
            f.write(f"{result}\n")


if __name__ == '__main__':
    main()