from utils import *
import torch.utils.data as data
import time

babyrobot_mapper = {
    "Happiness": 0,
    "Sadness": 1,
    "Surprise": 2,
    "Fear": 3,
    "Disgust": 4,
    "Anger": 5,
}


def get_db_splits(db):
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21],
            [22, 23, 24], [25, 26, 27], [28, 29, 30]]


def get_all_db_subjects(db):
    """ get  subjects for each db """
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30]


def get_babyrobot_annotations():
    unique_subjects = []
    subject_to_number = {}
    data = []

    subj_idx = 0
    with open("BRED_dataset/annotations.csv") as data_file:
        for x in data_file.readlines()[1:]:
            v = {
                "path": x.split(",")[0].split(".")[0],
                "subject": x.split(",")[0].split("/")[2],
                "emotion": babyrobot_mapper[x.split(",")[1].strip()],  # map emotion to number
                "ann_1_does_emotion_body": x.split(",")[2].strip(),
                "ann_1_does_emotion_face": x.split(",")[3].strip(),
                "ann_2_does_emotion_body": x.split(",")[4].strip(),
                "ann_2_does_emotion_face": x.split(",")[5].strip(),
                "ann_3_does_emotion_body": x.split(",")[6].strip(),
                "ann_3_does_emotion_face": x.split(",")[7].strip(),
            }

            l = [v['ann_1_does_emotion_face'], v['ann_2_does_emotion_face'], v['ann_3_does_emotion_face']]
            v['does_emotion_face'] = max(set(l), key=l.count)
            l = [v['ann_1_does_emotion_body'], v['ann_2_does_emotion_body'], v['ann_3_does_emotion_body']]
            v['does_emotion_body'] = max(set(l), key=l.count)
            data.append(v)
            subject = v['subject']

            if subject not in unique_subjects:
                unique_subjects.append(subject)
                subject_to_number[subject] = subj_idx
                subj_idx += 1

    return data, subject_to_number


class BodyFaceDataset(data.Dataset):
    def __init__(self, args, data=None, indices=None, subjects=None, phase=None):
        self.args = args
        self.phase = phase
        if args.db == "babyrobot":
            face, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data
            self.face = [face[x] for x in indices]
            self.bodies = [bodies[x] for x in indices]
            self.hands_right = [hands_right[x] for x in indices]
            self.hands_left = [hands_left[x] for x in indices]
            self.lengths = [lengths[x] for x in indices]
            self.Y = [Y[x] for x in indices]
            self.Y_face = [Y_face[x] for x in indices]
            self.Y_body = [Y_body[x] for x in indices]
            self.paths = [paths[x] for x in indices]
            self.groups = [groups[x] for x in indices]

        self.lengths = []
        for index in range(len(self.bodies)):
            self.lengths.append(self.bodies[index].shape[0])

        self.features = []
        for index in range(len(self.bodies)):
            features_path = self.paths[index] + "/cnn_features"
            features = torch.load(features_path, map_location=lambda storage, loc: storage)
            self.features.append(features)

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.hands_right = [scaler['hands_right'].transform(x) for x in self.hands_right]
        self.hands_left = [scaler['hands_left'].transform(x) for x in self.hands_left]
        self.bodies = [scaler['bodies'].transform(x) for x in self.bodies]
        self.face = [scaler['face'].transform(x) for x in self.face]

    def to_tensors(self):
        self.hands_right = [torch.from_numpy(x).float() for x in self.hands_right]
        self.hands_left = [torch.from_numpy(x).float() for x in self.hands_left]
        self.bodies = [torch.from_numpy(x).float() for x in self.bodies]
        self.face = [torch.from_numpy(x).float() for x in self.face]

    def prepad(self):
        """ prepad sequences to the max length sequence of each database """
        max_len = 323
        self.bodies = pad_sequence(self.bodies, batch_first=True, max_len=max_len)
        self.hands_right = pad_sequence(self.hands_right, batch_first=True, max_len=max_len)
        self.hands_left = pad_sequence(self.hands_left, batch_first=True, max_len=max_len)
        # self.faces = pad_sequence(self.faces, batch_first=True, max_len=max_len)
        self.face = pad_sequence(self.face, batch_first=True, max_len=max_len)
        self.features = pad_sequence(self.features, batch_first=True, max_len=max_len)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        v = time.time()
        body = self.bodies[index]
        hand_right = self.hands_right[index]
        hand_left = self.hands_left[index]
        # faces = self.faces[index]
        face = self.face[index]
        length = self.lengths[index]

        if self.args.use_cnn_features:
            features = self.features[index]
        else:
            features = torch.Tensor(1)

        if self.args.db == "babyrobot":
            label_face = self.Y_face[index]
            label_body = self.Y_body[index]

        return {
            "face": face,
            "body": body,
            "hand_left": hand_left,
            "hand_right": hand_right,
            "label": self.Y[index],
            "label_face": label_face,
            "label_body": label_body,
            "length": length,
            "paths": self.paths[index],
            "facial_cnn_features": features
        }
