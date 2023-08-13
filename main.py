import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import argparse
from sklearn.metrics import precision_recall_fscore_support
import csv
from datasets import *
from models import *
from utils import *
import warnings
warnings.filterwarnings("ignore")


class EmotionRecognitionSystem():
    def __init__(self, args):
        print(args)
        self.args = args

    def run(self):
        args = self.args
        self.all_iteration_accuracies = []

        all_iterations_accuracy_meter_top_all = []
        all_iterations_accuracy_meter_top_face = []
        all_iterations_accuracy_meter_top_body = []
        # SK Prec、 SK Rec、 F-Score
        all_iterations_p = []
        all_iterations_r = []
        all_iterations_f = []

        for i in range(self.args.num_total_iterations):
            self.current_iteration = i
            val_top_all, val_top_body, val_top_face, p, r, f = self.cross_validation(num_splits=args.num_splits)

            all_iterations_accuracy_meter_top_all.append(val_top_all)
            all_iterations_accuracy_meter_top_face.append(val_top_face)
            all_iterations_accuracy_meter_top_body.append(val_top_body)
            all_iterations_p.append(p)
            all_iterations_r.append(r)
            all_iterations_f.append(f)
            print('4-10 [Iteration: %02d/%02d] Top1 Accuracy: %.3f Accuracy Body %.3f Accuracy Face '
                  '%.3f SK Prec: %.3f, SK Rec: %.3f F-Score: %.3f' % (i + 1, args.num_total_iterations,
                                                                      np.mean(all_iterations_accuracy_meter_top_all),
                                                                      np.mean(all_iterations_accuracy_meter_top_body),
                                                                      np.mean(all_iterations_accuracy_meter_top_face),
                                                                      np.mean(all_iterations_p),
                                                                      np.mean(all_iterations_r),
                                                                      np.mean(all_iterations_f)))

            csv_file_all = csv.writer(open('data_all.csv', 'a', newline=''))
            data_all_csv_data = [[np.mean(all_iterations_accuracy_meter_top_all),
                                  np.mean(all_iterations_accuracy_meter_top_body),
                                  np.mean(all_iterations_accuracy_meter_top_face),
                                  np.mean(all_iterations_p),
                                  np.mean(all_iterations_r),
                                  np.mean(all_iterations_f)]]
            csv_file_all.writerows(data_all_csv_data)

    def get_scaler(self):
        scaler = {}
        feats = ["bodies", "face", "hands_right", "hands_left", ]

        for x in feats:
            all_data = np.vstack(getattr(self.train_dataset, x))
            scaler[x] = MinMaxScaler()
            scaler[x].fit(all_data)

        return scaler

    def cross_validation(self, num_splits):
        cross_val_accuracy_meter_top_all = []
        cross_val_accuracy_meter_top_face = []
        cross_val_accuracy_meter_top_body = []

        cross_val_p = []
        cross_val_r = []
        cross_val_f = []
        data_dic, subject_to_number = get_babyrobot_annotations()
        """
        data:list[]
        face: list, each openpose is one[], each [] have jsons in openpose. have 215 []   (82, 210),(87,210),(92,210)...
        bodies: list,  (82,75),(87,75),(92,75)...
        hands_left,hand_right: list,  (82,63),(87,63),(92,63)...
        lengths: list, is the number of video frames   (82, 87, 92...)
        Y: int(emotion), emotion number   (5, 4, 3, 0...)
        Y_face: int(emotion) face emotion number  (5, 4, 6, 0...)
        Y_body: int(emotion) body emotion number   (5, 6, 3, 6...)
        paths:  (...Anger, ...Disgust, ....)
        groups: (0, 0, 0, 0, 1 ,1, ....)
        """
        face, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data
        self.kfold = GroupKFold(n_splits=num_splits).split(bodies, Y_body, groups)

        splits_pic = []
        for n in range(num_splits):
            self.current_split = n
            splits_pic.append(self.current_split)
            train_idx, test_idx = next(self.kfold)
            self.train_dataset = BodyFaceDataset(data=data, indices=train_idx, phase="train", args=self.args)
            self.test_dataset = BodyFaceDataset(data=data, indices=test_idx, phase="val", args=self.args)

            print("train samples: %d" % len(self.train_dataset))
            print("all: number of labels for each classes:", np.bincount(self.train_dataset.Y))
            if self.args.db == "babyrobot":
                print("face: number of labels for each classes:", np.bincount(self.train_dataset.Y_face))
                print("body: number of labels for each classes:", np.bincount(self.train_dataset.Y_body))

            print("test samples: %d" % len(self.test_dataset))
            print("all: number of labels for each classes:", np.bincount(self.test_dataset.Y))
            if self.args.db == "babyrobot":
                print("face: number of labels for each classes:", np.bincount(self.test_dataset.Y_face))
                print("body: number of labels for each classes:", np.bincount(self.test_dataset.Y_body))

            scaler = self.get_scaler()
            self.train_dataset.set_scaler(scaler)
            self.test_dataset.set_scaler(scaler)
            self.train_dataset.to_tensors()
            self.test_dataset.to_tensors()
            self.train_dataset.prepad()
            self.test_dataset.prepad()

            print("scaled data")
            if self.args.batch_size == -1:
                batch_size = len(self.train_dataset)
            else:
                batch_size = self.args.batch_size

            self.dataloader_train = torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size,
                                                                drop_last=True, num_workers=0)
            self.dataloader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset),
                                                               num_workers=0)

            self.model = BodyFaceEmotionClassifier(self.args, 323).cuda()
            start = time.time()
            val_top_all, val_top_body, val_top_face, p, r, f = self.fit(self.model, n)
            end = time.time()
            cross_val_accuracy_meter_top_all.append(val_top_all)
            cross_val_accuracy_meter_top_body.append(val_top_body)
            cross_val_accuracy_meter_top_face.append(val_top_face)

            cross_val_p.append(p)
            cross_val_r.append(r)
            cross_val_f.append(f)

            print('[Split: %02d/%02d] Accuracy: %.3f Body Accuracy: %.3f Face Accuracy: %.3f SK '
                  'Prec: %.3f SK Rec: %.3f F-Score: %.3f Time: %.3f' % (n + 1, num_splits, np.mean(cross_val_accuracy_meter_top_all),
                     np.mean(cross_val_accuracy_meter_top_body), np.mean(cross_val_accuracy_meter_top_face),
                     np.mean(cross_val_p), np.mean(cross_val_r), np.mean(cross_val_f), end - start))

            csv_file_val = csv.writer(open('val.csv', 'a', newline=''))
            val_to_csv_data = [[n+1, np.mean(cross_val_accuracy_meter_top_all),
                                np.mean(cross_val_accuracy_meter_top_body),
                                np.mean(cross_val_accuracy_meter_top_face), np.mean(cross_val_p), np.mean(cross_val_r),
                                np.mean(cross_val_f), end - start]]
            csv_file_val.writerows(val_to_csv_data)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(splits_pic, cross_val_accuracy_meter_top_all, label='all')
        ax.plot(splits_pic, cross_val_accuracy_meter_top_body, label='body')
        ax.plot(splits_pic, cross_val_accuracy_meter_top_face, label='face')
        ax.set_xlabel('10split')
        ax.set_ylabel('accuracy')
        ax.set_title('val accuracy')
        ax.legend()
        plt.savefig('val_acc.png', dpi=200)
        plt.show()
        plt.close()

        return np.mean(cross_val_accuracy_meter_top_all), np.mean(cross_val_accuracy_meter_top_body), np.mean(
            cross_val_accuracy_meter_top_face), np.mean(cross_val_p), np.mean(cross_val_r), np.mean(cross_val_f)

    def fit(self, model, n):
        if self.args.weighted_loss:
            if self.args.split_branches:
                self.criterion_both = nn.CrossEntropyLoss().cuda()
                self.criterion_face = nn.CrossEntropyLoss().cuda()
                self.criterion_body = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(get_weighted_loss_weights(self.train_dataset.Y_body, 6))).cuda()
            elif self.args.use_labels == "body":
                self.criterion = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(get_weighted_loss_weights(self.train_dataset.Y_body, 6))).cuda()
            else:
                self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            if self.args.split_branches:
                self.criterion_both = nn.CrossEntropyLoss().cuda()
                self.criterion_face = nn.CrossEntropyLoss().cuda()
                self.criterion_body = nn.CrossEntropyLoss().cuda()
            elif self.args.use_labels == "body":
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss().cuda()

        if self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.1)

        best_acc = 0
        train_acc_pic = []
        train_loss_pic = []
        epochs_pic = []
        val_acc_all = []
        val_body_acc_all = []
        val_face_acc_all = []
        p_top = 0
        r_top = 0
        f_top = 0

        for current_epoch in range(0, self.args.epochs):
            epochs_pic.append(current_epoch)
            train_acc, train_loss = self.train_epoch()
            train_acc_pic.append(float(train_acc))
            train_loss_pic.append(float(train_loss))
            
            val_all, val_body, val_face, p, r, f = self.eval(n)
            val_acc_all.append(float(val_all))
            val_body_acc_all.append(float(val_body))
            val_face_acc_all.append(float(val_face))
            
            if p > p_top:
              p_top = p
            if r > r_top:
              r_top = r
            if f > f_top:
              f_top = f

            csv_file_each_train = csv.writer(open('train_each.csv', 'a', newline=''))
            each_train_to_csv_data = [[n, current_epoch, train_loss, train_acc]]
            csv_file_each_train.writerows(each_train_to_csv_data)
            
            csv_file_each_val = csv.writer(open('val_each.csv', 'a', newline=''))
            each_val_to_csv_data = [[n, current_epoch, val_all, val_body, val_face, p, r, f]]
            csv_file_each_val.writerows(each_val_to_csv_data)

            if current_epoch == self.args.epochs - 1:
                val_top = np.max(val_acc_all)
                val_top_body = np.max(val_body_acc_all)
                val_top_face = np.max(val_face_acc_all)

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f, Validation Acc: %.3f, '
                  'Validation Acc Body: %.3f, Validation Acc Face: %.3f, SK Prec: %.3f, SK Rec: %.3f, '
                  'F-Score: %.3f, Learning Rate:%.8f' % (current_epoch, self.args.epochs, train_loss,
                                                         train_acc, val_all, val_body, val_face, p, r, f, lr))

            if val_all > best_acc:
                best_acc = val_all
                torch.save(model.state_dict(), "./weights/model-{}.pth".format(current_epoch))

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(epochs_pic, train_loss_pic, label='train_loss')
        ax.plot(epochs_pic, train_acc_pic, label='train_acc')
        ax.plot(epochs_pic, val_acc_all, label='val_acc')
        ax.plot(epochs_pic, val_body_acc_all, label='val_body_acc')
        ax.plot(epochs_pic, val_face_acc_all, label='val_face_acc')
        ax.set_xlabel('epoch')
        ax.set_ylabel('acc & loss')
        ax.set_title('training&val')
        ax.legend()
        plt.savefig('each_train&val_{}.png'.format(n), dpi=200)
        plt.show()
        plt.close()

        return val_top, val_top_body, val_top_face, p, r, f

    def train_epoch(self):
        global acc_meter_top_all
        self.model.train()

        accuracy_meter_top_all = AverageMeter()
        loss_meter = AverageMeter()

        for i, batch in enumerate(self.dataloader_train):
            facial_cnn_features, face, body, hand_right, hand_left, length, y, y_face, y_body = \
                batch['facial_cnn_features'].cuda(), batch['face'].cuda(), batch['body'].cuda(), \
                batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch['length'].cuda(), batch['label'].cuda(), \
                batch['label_face'].cuda(), batch['label_body'].cuda()

            self.optimizer.zero_grad()
            if self.args.split_branches:
                if self.args.do_fusion:
                    out_face, out_body, out_fusion = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features), self.args.batch_size)
                    loss = 0
                    loss_fusion = self.criterion_both(out_fusion, y)
                    loss_body = self.criterion_body(out_body, y_body)
                    loss_face = self.criterion_face(out_face, y_face)
                    loss = loss_body + loss_face + loss_fusion
                    loss.backward()
                else:
                    out, out_body, out_face = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features))
                    loss_total = self.criterion_both(out, y)
                    loss_body = self.criterion_body(out_body, y_body)
                    loss_face = self.criterion_face(out_face, y_face)
                    loss = loss_body + loss_face + loss_total
                    loss.backward()
            else:
                out = self.model.forward(
                    (face, body, hand_right, hand_left, length, facial_cnn_features), self.args.batch_size)

                if self.args.use_labels == "body":
                    loss = self.criterion(out, y_body)
                elif self.args.use_labels == "face":
                    loss = self.criterion(out, y_face)
                else:
                    loss = self.criterion(out, y)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            if self.args.split_branches:
                if args.do_fusion:
                    accs = accuracy(out_fusion, y, topk=(1,))
                else:
                    accs = accuracy(out, y, topk=(1,))
                accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                loss_meter.update(loss.item(), length.size(0))
            else:
                if self.args.use_labels == "body":
                    accs = accuracy(out, y_body, topk=(1,))
                elif self.args.use_labels == "face":
                    accs = accuracy(out, y_face, topk=(1,))
                else:
                    accs = accuracy(out, y, topk=(1,))
                accuracy_meter_top_all.update(accs[0], body.size(0))
                loss_meter.update(loss.item(), body.size(0))

        self.scheduler.step()
        return accuracy_meter_top_all.avg, loss_meter.avg

    def eval(self, num):
        accuracy_meter_top_all = AverageMeter()
        accuracy_meter_top_face = AverageMeter()
        accuracy_meter_top_body = AverageMeter()

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.dataloader_test):
                facial_cnn_features, face, body, hand_right, hand_left, length, y, y_face, y_body = \
                    batch['facial_cnn_features'].cuda(), batch['face'].cuda(), batch['body'].cuda(), \
                    batch['hand_right'].cuda(), batch['hand_left'].cuda(),\
                    batch['length'].cuda(), batch['label'].cuda(), batch['label_face'].cuda(), batch['label_body'].cuda()
                if self.args.split_branches:
                    if self.args.do_fusion:
                        out_face, out_body, out_fusion = self.model.forward(
                            (face, body, hand_right, hand_left, length,
                             facial_cnn_features), self.args.batch_size)

                        accs = accuracy(out_fusion, y, topk=(1,))
                        accs_face = accuracy(out_face, y_face, topk=(1,))
                        accs_body = accuracy(out_body, y_body, topk=(1,))

                        accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                        accuracy_meter_top_body.update(accs_body[0].item(), length.size(0))
                        accuracy_meter_top_face.update(accs_face[0].item(), length.size(0))

                        p, r, f, s = precision_recall_fscore_support(y.cpu(), out_fusion.detach().cpu().argmax(dim=1),
                                                                     average="macro")
                    else:
                        out_body, out_face = self.model.forward(
                            (face, body, hand_right, hand_left, length, facial_cnn_features), self.args.batch_size)
                        accs = accuracy(out, y, topk=(1,))
                        accs_face = accuracy(out_face, y_face, topk=(1,))
                        accs_body = accuracy(out_body, y_body, topk=(1,))
                        accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                        accuracy_meter_top_body.update(accs_body[0].item(), length.size(0))
                        accuracy_meter_top_face.update(accs_face[0].item(), length.size(0))
                        p, r, f, s = precision_recall_fscore_support(y.cpu(), out.detach().cpu().argmax(dim=1),
                                                                     average="macro")
                else:
                    out = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features), self.args.batch_size)
                    if self.args.use_labels == "body":
                        t = y
                        y = y_body
                    elif self.args.use_labels == "face":
                        t = y
                        y = y_face
                    accs = accuracy(out, y, topk=(1,))
                    """ change average to the desired (macro for balanced) """
                    p, r, f, s = precision_recall_fscore_support(y.cpu(), out.detach().cpu().argmax(dim=1),
                                                                 average="macro")
                    accuracy_meter_top_all.update(accs[0].item(), length.size(0))

        return accuracy_meter_top_all.avg, accuracy_meter_top_body.avg, accuracy_meter_top_face.avg, p * 100, r * 100, f * 100


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    # ========================= Optimizer Parameters ==========================
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    # ========================= Usual Hyper Parameters ==========================
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--db', default="babyrobot")
    parser.add_argument('--exp_name', default="HMT-4")
    parser.add_argument('--epochs', default=130, type=int)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    # ========================= Network Parameters ==========================
    parser.add_argument('--do_fusion', action="store_true", dest="do_fusion", default=True,
                        help="do the final fusion of face, body and whole body emotion scores")
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--use_cnn_features', action="store_true", dest="use_cnn_features", default=True,
                        help="add features from affectnet cnn")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--num_total_iterations', type=int, default=1)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--add_body_dnn', action="store_true", dest="add_body_dnn", default=True,
                        help="use a dnn for modeling the skeleton")
    parser.add_argument('--first_layer_size', default=256, type=int)
    parser.add_argument('--all_features', action="store_true", dest="all_features", default=True,
                        help="add the all(face and body) keypoints branch")

    parser.add_argument('--cross_attention', action='store_true', dest='do_cross_attention',
                        default=True, help="do cross attention")
    parser.add_argument('--adapt_fusion', action='store_true', dest='do_adapt_fusion',
                        default=False, help="do adaptive feature fusion")
    # ========================= Training Parameters ==========================
    parser.add_argument('--split_branches', action="store_true", dest="split_branches", default=True,
                        help="split emotion calculations of face and body (hierarchical labels training)")
    parser.add_argument('--add_whole_body_branch', action="store_true", dest="add_whole_body_branch", default=False,
                        help="how to fuse face-body in the whole body branch")
    parser.add_argument('--weighted_loss', action="store_true", dest="weighted_loss", default=True)  # use weighted loss
    parser.add_argument('--use_labels', type=str,
                        help="if you want to train only body or face models, select 'body' or 'face'")

    args = parser.parse_args()
    return args


args = parse_opts()
# model = BodyFaceEmotionClassifier(args, 80)
# print(model)
b = EmotionRecognitionSystem(args)
b.run()
