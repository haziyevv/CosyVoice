import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    wavs = list(glob.glob('{}/*.wav'.format(args.src_dir)))
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    speaker = 'Twilight-Sparkle'  # Fixed speaker ID for all utterances
    spk2utt[speaker] = []  # Initialize the list for our single speaker

    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.txt')  # Changed from .normalized.txt to .txt
        if not os.path.exists(txt):
            logger.warning('{} does not exist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.strip() for l in f.readlines())  # Changed to handle multiple lines
        utt = os.path.basename(wav).replace('.wav', '')
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2spk[utt] = speaker
        spk2utt[speaker].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()