import mne
from pyprep.find_noisy_channels import NoisyChannels
import argparse
import os


def get_input_path(subject_num, session_num):
    subject_str = f'sub-{subject_num:02d}'
    session_str = f'ses-{session_num:02d}'
    input_path = os.path.join('Dataset', subject_str, session_str, 'eeg', f"{subject_str}_{session_str}_eeg.fif")
    return input_path

def get_output_path(subject_num, session_num):
    subject_str = f'sub-{subject_num:02d}'
    session_str = f'ses-{session_num:02d}'
    output_dir = os.path.join('Dataset', 'derivatives', 'preprocessed_eeg', subject_str, session_str, 'eeg')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{subject_str}_{session_str}_eeg.fif")
    return output_path



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess EEG .edf data.')
    parser.add_argument('--subject-num', type=int, required=True, help='Subject id (integer) starts from 1')
    parser.add_argument('--session-num', type=int, required=True, help='Session id (integer) starts from 1')
    parser.add_argument("--gender", type=str, choices=["male", "female"], required=True, help="Subject gender.")

    args = parser.parse_args()

    SUBJECT_NUM = args.subject_num
    SESSION_NUM = args.session_num
    SUBJECT_GENDER = args.gender


    input_path = get_input_path(SUBJECT_NUM, SESSION_NUM)
    output_path = get_output_path(SUBJECT_NUM, SESSION_NUM)
    print("input .edf path:", input_path)
    print("save .fif path:", output_path)
    print("input gender:", SUBJECT_GENDER)
    
    # mne_raw_eeg = mne.io.read_raw_edf(input_path, preload = True)
    mne_raw_eeg = mne.io.read_raw_fif(input_path, preload = True)

    mne_raw_eeg.set_montage('standard_1020')

    # 1. Notch filter
    mne_raw_eeg.notch_filter(freqs = [50, 100])
    print("Notch at 50Hz and 100Hz")

    # 2. Bandpass filter: low_cut = 3 Hz, high_cut = 45 Hz
    SAMPLE_RATE = int(mne_raw_eeg.info['sfreq'])
    IMAGINE_DURATION_SEC = 2
    mne_raw_eeg.filter(3, 45, l_trans_bandwidth = 2, h_trans_bandwidth = 2, filter_length = (SAMPLE_RATE * IMAGINE_DURATION_SEC) )
    print("Bandpass filter low_cut = 3 Hz, high_cut = 45 Hz")


    # 3. Detect bad channels
    if SUBJECT_GENDER == "female":
        # For female subjects the Fz channel was used as reference so the Fz electorode is not attached
        # We mark it as bad channel to interpolate it.
        mne_raw_eeg.info['bads'] += ['Fz']
        print("Female subject --> marked Fz as bad channel")

    find_noisy_channels = NoisyChannels(mne_raw_eeg, do_detrend = False, random_state = 1)
    find_noisy_channels.find_bad_by_correlation(frac_bad = 0.02)
    mne_raw_eeg.info['bads'] += find_noisy_channels.get_bads()

    print("Marked bad channels:", mne_raw_eeg.info['bads'])


    # 4. Interpolate bad channels
    mne_raw_eeg.interpolate_bads(reset_bads=True)
    assert len(mne_raw_eeg.info['bads']) == 0, "Error : Interpolation problem, bad channels still exist!"
    print("Bad channels after interpolate:", mne_raw_eeg.info['bads'])


    # 5. Re-referencing
    mne_raw_eeg.set_eeg_reference(ref_channels='average')

    print(mne_raw_eeg.get_data())

    mne_raw_eeg.save(output_path, overwrite = True)
    print("🟢 Saved preprocessed signal to", output_path)


if __name__ == "__main__":
    main()
