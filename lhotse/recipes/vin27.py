"""
This recipe supports corpora: Vin27.

---

"""
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import json
import os 
from tqdm import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, resumable_download, safe_extract

# LIBRITTS = (
#     "dev-clean",
#     "dev-other",
#     "test-clean",
#     "test-other",
#     "train-clean-100",
#     "train-clean-360",
#     "train-other-500",
# )


def prepare_vin27(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None
    # num_jobs: int = 1,
    # link_previous_utt: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :param link_previous_utt: If true adds previous utterance id to supervisions.
        Useful for reconstructing chains of utterances as they were read.
        If previous utterance was skipped from LibriTTS datasets previous_utt label is None.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    # corpus_dir = Path(corpus_dir)
    assert Path(corpus_dir).is_dir(), f"No such directory: {Path(corpus_dir)}"

    # if dataset_parts == "all" or dataset_parts[0] == "all":
    #     dataset_parts = LIBRITTS
    # elif isinstance(dataset_parts, str):
    #     assert dataset_parts in LIBRITTS
    #     dataset_parts = [dataset_parts]

    # manifests = {}
    list_vietnamese_north_provinces = "bac-giang bac-kan bac-ninh cao-bang ha-giang ha-nam ha-noi ha-tay hai-duong hai-phong hoa-binh hung-yen lai-chau lang-son lao-cai nam-đinh ninh-binh phu-tho quang-ninh son-la thai-binh thai-nguyen tuyen-quang vinh-phuc yen-bai đien-bien".split(" ")
    list_vietnamese_center_provinces = "binh-thuan binh-đinh gia-lai ha-tinh khanh-hoa kon-tum lam-đong nghe-an ninh-thuan phu-yen quang-binh quang-nam quang-ngai quang-tri thanh-hoa thua-thien---hue đa-nang đak-lak đak-nong".split(" ")
    list_vietnamese_south_provinces = "an-giang ba-ria-vung-tau bac-lieu ben-tre binh-duong binh-phuoc ca-mau can-tho hau-giang ho-chi-minh kien-giang long-an soc-trang tay-ninh tien-giang tp.-ho-chi-minh tra-vinh vinh-long đong-nai đong-thap".split(" ")
    
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        # manifests = read_manifests_if_cached(
        #     dataset_parts=dataset_parts, output_dir=output_dir, prefix="libritts"
        # )

    # Contents of the file
    #   ;ID  |SEX| SUBSET           |MINUTES| NAME
    #   14   | F | train-clean-360  | 25.03 | ...
    #   16   | F | train-clean-360  | 25.11 | ...
    #   17   | M | train-clean-360  | 25.04 | ...
    spk2gender = {}
    dictionary_transcription = {}
    if (Path(corpus_dir) / "transcription.json").is_file():
        with open(Path(corpus_dir) / "transcription.json", "r") as file:
            dictionary_transcription = json.load(file)
    vietnamese_syllables = []
    if (Path(corpus_dir) / "vietnamese_syllables.txt").is_file():
        with open(Path(corpus_dir) / "vietnamese_syllables.txt", "r", encoding="utf-16") as file:
            vietnamese_syllables = file.read().split("\n")
    vietnamese_syllables = set(vietnamese_syllables)
    
    recordings_train = []
    supervisions_train = []
    
    recordings_dev = []
    supervisions_dev = []
    recordings_test = []
    supervisions_test = []
    for province in dictionary_transcription.keys():
        print(province)
        list_speakers_train = []
        list_speakers_dev = []
        list_speakers_test = []
        for speaker in tqdm(dictionary_transcription[province].keys()):
            if len(list_speakers_test) < 1:
                list_speakers_test.append(speaker)
            elif len(list_speakers_dev) < 1:
                list_speakers_dev.append(speaker)
            # elif len(list_speakers_train) < int(0.05 * len(dictionary_transcription[province].keys())):
            #     list_speakers_train.append(speaker)
            # else:
            #     break
            else: 
                list_speakers_train.append(speaker)
            print("Speaker: ", speaker)
            
            for wav in dictionary_transcription[province][speaker].keys():
                audio_path = corpus_dir + "/" + province + "/" + speaker + "/" + wav
                ###
                orgin_text = ""
                if dictionary_transcription[province][speaker][wav]["origin_text"][-1] == ".":
                    orgin_text = dictionary_transcription[province][speaker][wav]["origin_text"][:-1]
                list_syls = orgin_text.lower().split(" ")
                if not set(list_syls).issubset(vietnamese_syllables):
                    continue
                
                if not os.path.isfile(audio_path):
                    logging.warning(f"No such file: {audio_path}")
                    continue
                rec_id = province + "_" + speaker + "_" + wav
                print("Rec_id: ", rec_id)
                recording = Recording.from_file(audio_path, recording_id=rec_id)
                if province in list_vietnamese_south_provinces:
                    locale = "nam"
                elif province in list_vietnamese_center_provinces:
                    locale = "trung"
                elif province in list_vietnamese_north_provinces:
                    locale = "bac"
                segment = SupervisionSegment(
                    id=rec_id,
                    recording_id=rec_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    text=dictionary_transcription[province][speaker][wav]["normalized_text"],
                    language="Vietnamese",
                    speaker=province + "_" + speaker,
                    gender=dictionary_transcription[province][speaker][wav]["gender"],
                    custom={"orig_text": dictionary_transcription[province][speaker][wav]["origin_text"], "locale": locale},
                )
                if speaker in list_speakers_test:
                    recordings_test.append(recording)
                    supervisions_test.append(segment)
                elif speaker in list_speakers_dev:
                    recordings_dev.append(recording)
                    supervisions_dev.append(segment)
                elif speaker in list_speakers_train:
                    recordings_train.append(recording)
                    supervisions_train.append(segment)
    number_of_parts = 10
    assert len(recordings_train) == len(supervisions_train)
    dictionary_recordings_train = {}
    dictionary_supervisions_train = {}
    
    number_of_items_per_part = int(len(recordings_train) / number_of_parts)
    for i in range(number_of_parts):
        st = i * number_of_items_per_part
        end = (i+1) * number_of_items_per_part
        if i != (number_of_parts - 1):
            recording_set_train = RecordingSet.from_recordings(recordings_train[st:end])
            supervision_set_train = SupervisionSet.from_segments(supervisions_train[st:end])
        else:
            recording_set_train = RecordingSet.from_recordings(recordings_train[st:])
            supervision_set_train = SupervisionSet.from_segments(supervisions_train[st:])
        recording_set_train, supervision_set_train = fix_manifests(recording_set_train, supervision_set_train)
        validate_recordings_and_supervisions(recording_set_train, supervision_set_train)
        dictionary_recordings_train[i] = recording_set_train
        dictionary_supervisions_train[i] = supervision_set_train
                        
    # recording_set_train = RecordingSet.from_recordings(recordings_train)
    # supervision_set_train = SupervisionSet.from_segments(supervisions_train)
    # recording_set_train, supervision_set_train = fix_manifests(recording_set_train, supervision_set_train)
    # validate_recordings_and_supervisions(recording_set_train, supervision_set_train)
    
    recording_set_dev = RecordingSet.from_recordings(recordings_dev)
    supervision_set_dev = SupervisionSet.from_segments(supervisions_dev)
    recording_set_dev, supervision_set_dev = fix_manifests(recording_set_dev, supervision_set_dev)
    validate_recordings_and_supervisions(recording_set_dev, supervision_set_dev)
    
    recording_set_test = RecordingSet.from_recordings(recordings_test)
    supervision_set_test = SupervisionSet.from_segments(supervisions_test)
    recording_set_test, supervision_set_test = fix_manifests(recording_set_test, supervision_set_test)
    validate_recordings_and_supervisions(recording_set_test, supervision_set_test)
    
    output = {}
    
    if output_dir is not None:
        for i in dictionary_recordings_train.keys():
            out_record_file =  "vin27_recordings_train_part_" + str(i) + ".jsonl.gz"
            out_sup_file =  "vin27_supervisions_train_part_" + str(i) + ".jsonl.gz" 
            dictionary_supervisions_train[i].to_file(output_dir / out_sup_file)
            dictionary_recordings_train[i].to_file(output_dir / out_record_file)
            output["train_part_" + str(i)] = {"recordings": dictionary_recordings_train[i], "supervisions": dictionary_supervisions_train[i]}
        #
        supervision_set_dev.to_file(output_dir / "vin27_supervisions_dev.jsonl.gz")
        recording_set_dev.to_file(output_dir / "vin27_recordings_dev.jsonl.gz")
        #
        supervision_set_test.to_file(output_dir / "vin27_supervisions_test.jsonl.gz")
        recording_set_test.to_file(output_dir / "vin27_recordings_test.jsonl.gz")

        output["dev"] = {"recordings": recording_set_dev, "supervisions": supervision_set_dev}
        output["test"] = {"recordings": recording_set_test, "supervisions": supervision_set_test}
    return output

# def feature_extractor() -> TorchaudioFeatureExtractor:
#     """
#     Set up the feature extractor for TTS task.
#     :return: A feature extractor with custom parameters.
#     """
#     feature_extractor = Fbank()
#     feature_extractor.config.num_mel_bins = 80

#     return feature_extractor


# def text_normalizer(segment: SupervisionSegment) -> SupervisionSegment:
#     text = segment.text.upper()
#     text = re.sub(r"[^\w !?]", "", text)
#     text = re.sub(r"^\s+", "", text)
#     text = re.sub(r"\s+$", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return fastcopy(segment, text=text)
    # for part in tqdm(dataset_parts, desc="Preparing LibriTTS parts"):
    #     if manifests_exist(part=part, output_dir=output_dir, prefix="libritts"):
    #         logging.info(f"LibriTTS subset: {part} already prepared - skipping.")
    #         continue
    #     part_path = corpus_dir / part
    #     # We are ignoring weird files such as ._84_121550_000007_000000.wav
    #     # Maybe LibriTTS-R will fix it in later distributions.
    #     # Also, the file 1092_134562_000013_000004.wav is corrupted as of May 31st.
    #     recordings = RecordingSet.from_dir(
    #         part_path,
    #         "*.wav",
    #         num_jobs=num_jobs,
    #         exclude_pattern=r"^(\._.+|1092_134562_000013_000004\.wav)$",
    #     )
    #     supervisions = []
    #     for trans_path in tqdm(
    #         part_path.rglob("*.trans.tsv"),
    #         desc="Scanning transcript files (progbar per speaker)",
    #         leave=False,
    #     ):
    #         if re.match(r"^\._.+$", trans_path.name) is not None:
    #             continue
    #         # The trans.tsv files contain only the recordings that were kept for LibriTTS.
    #         # Example path to a file:
    #         #   /export/corpora5/LibriTTS/dev-clean/84/121123/84_121123.trans.tsv
    #         #
    #         # Example content:
    #         #   84_121123_000007_000001 Maximilian.     Maximilian.
    #         #   84_121123_000008_000000 Villefort rose, half ashamed of being surprised in such a paroxysm of grief.    Villefort rose, half ashamed of being surprised in such a paroxysm of grief.

    #         # book.tsv contains additional metadata
    #         utt2snr = [
    #             (rec_id, float(snr))
    #             for rec_id, *_, snr in map(
    #                 str.split,
    #                 (
    #                     trans_path.parent
    #                     / trans_path.name.replace(".trans.tsv", ".book.tsv")
    #                 )
    #                 .read_text()
    #                 .splitlines(),
    #             )
    #         ]
    #         # keeps the order of uttids as they appear in book.tsv
    #         uttids = [r for r, _ in utt2snr]
    #         utt2snr = dict(utt2snr)

    #         if link_previous_utt:
    #             # Using the property of sorted keys to find previous utterance
    #             # The keys has structure speaker_book_x_y e.g. 1089_134691_000004_000001
    #             utt2prevutt = dict(zip(uttids + [None], [None] + uttids))

    #         prev_rec_id = None
    #         for line in trans_path.read_text().splitlines():
    #             rec_id, orig_text, norm_text = line.split("\t")
    #             if rec_id not in recordings:
    #                 logging.warning(
    #                     f"No recording exists for utterance id {rec_id}, skipping (in {trans_path})"
    #                 )
    #                 continue
    #             spk_id = rec_id.split("_")[0]
    #             customd = {"orig_text": orig_text, "snr": utt2snr.get(rec_id)}
    #             if link_previous_utt:
    #                 # all recordings ids should be in the book.csv
    #                 # but they are some missing e.g. 446_123502_000030_000003
    #                 prev_utt = utt2prevutt.get(rec_id, None)
    #                 # previous utterance has to be present in trans.csv - otherwise it was skipped
    #                 prev_utt = prev_utt if prev_utt == prev_rec_id else None
    #                 customd["prev_utt"] = prev_utt
    #                 prev_rec_id = rec_id
    #             supervisions.append(
    #                 SupervisionSegment(
    #                     id=rec_id,
    #                     recording_id=rec_id,
    #                     start=0.0,
    #                     duration=recordings[rec_id].duration,
    #                     channel=0,
    #                     text=norm_text,
    #                     language="English",
    #                     speaker=spk_id,
    #                     gender=spk2gender.get(spk_id),
    #                     custom=customd,
    #                 )
    #             )

    #     supervisions = SupervisionSet.from_segments(supervisions)
    #     recordings, supervisions = fix_manifests(recordings, supervisions)
    #     validate_recordings_and_supervisions(recordings, supervisions)

    #     if output_dir is not None:
    #         supervisions.to_file(output_dir / f"libritts_supervisions_{part}.jsonl.gz")
    #         recordings.to_file(output_dir / f"libritts_recordings_{part}.jsonl.gz")

    #     manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    # return manifests


# prepare_librittsr = prepare_libritts
