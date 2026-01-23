import argparse
import logging
import glob
import os
from tqdm import tqdm
import re
import pickle

with open('/workspace/CosyVoice-mine-current/examples/libritts/cosyvoice3/gemini_voice_tags.pkl', 'rb') as fr:
    gemini_voice_tags = pickle.load(fr)

with open('/workspace/CosyVoice-mine-current/examples/libritts/cosyvoice3/acctress_voice_tags.pkl', 'rb') as fr:
    acctress_voice_tags = pickle.load(fr)

with open('/workspace/CosyVoice-mine-current/examples/libritts/cosyvoice3/acctress_emotion_tags.pkl', 'rb') as fr:
    acctress_emotion_tags = pickle.load(fr)

with open('/workspace/CosyVoice-mine-current/examples/libritts/cosyvoice3/prompt_counts.pkl', 'rb') as fr:
    prompt_counts = pickle.load(fr)

logger = logging.getLogger()


def main():
    wavs = list(glob.glob('{}/*wav'.format(args.src_dir)))

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.txt')
        if not os.path.exists(txt):
            logger.warning('{} do not exsist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())

        utt = os.path.basename(wav).replace('.wav', '')
        spk = args.src_dir.split('/')[-1]

        try:
            prompt, content = content.split('<|endofprompt|>')
        except:
            print(content)
            continue
        if 'test' in spk:
            spk = spk.split('-test')[0]
            if spk in {'Achernar', 'Aoede', 'Autonoe', 'Despina', 'Erinome', 'Kore', 'Leda', 'Pulcherrima', 'Sulafat', 'Vindemiatrix', 'Zephyr'}:
                tags = re.findall('\[[\w\s\-]+\]', content)
                if any([x not in gemini_voice_tags for x in tags]):
                    continue
                emotions = prompt.split(', ')
                if any([prompt_counts.get(x.strip(), 1) < 50 for x in emotions]):
                    continue
                content = f"You are a helpfull assistant. You are {spk}. Speak in a {prompt.strip()} tone<|endofprompt|>{content}"

            else:
                tags = re.findall('\[[\w\s\-]+\]', content)
                if any([x not in acctress_voice_tags for x in tags]):
                    continue
                emotion_tags = re.findall('\<[\w\s\-]+\>', content)
                if any([tag not in acctress_emotion_tags for tag in emotion_tags]):
                    continue
                content = f"You are a helpfull assistant. You are {spk}. {prompt.strip()}<|endofprompt|>{content}"
        else:
            if spk in {'Achernar', 'Aoede', 'Autonoe', 'Despina', 'Erinome', 'Kore', 'Leda', 'Pulcherrima', 'Sulafat', 'Vindemiatrix', 'Zephyr'}:
                tags = re.findall('\[[\w\s\-]+\]', content)
                if any([x not in gemini_voice_tags for x in tags]):
                    continue
                emotions = prompt.split(', ')
                if any([prompt_counts.get(x.strip(), 1) < 50 for x in emotions]):
                    continue
                content = f"You are a helpfull assistant. You are {spk}. Speak in a {prompt.strip()} tone<|endofprompt|>{content}"

            else:
                tags = re.findall('\[[\w\s\-]+\]', content)
                if any([x not in acctress_voice_tags for x in tags]):
                    continue
                emotion_tags = re.findall('\<[\w\s\-]+\>', content)
                if any([tag not in acctress_emotion_tags for tag in emotion_tags]):
                    continue
                content = f"You are a helpfull assistant. You are {spk}. {prompt.strip()}<|endofprompt|>{content}"
        
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)
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
    if args.instruct is True:
        with open('{}/instruct'.format(args.des_dir), 'w') as f:
            for k, v in utt2text.items():
                # NOTE in CosyVoice3, we add instruct in sequence
                f.write('{} You are a helpful assistant.<|endofprompt|>\n'.format(k, v))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--instruct',
                        action='store_true',
                        default=False,
                        help='create instruct file or not')
    args = parser.parse_args()
    main()
