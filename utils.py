import torch
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import errno
import os


def accuracy(output, target, topk=(1,)):
    """Computes the precision @k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

plt.rcParams["figure.figsize"] = [16, 9]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_weighted_loss_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    labels_array = dataset

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_array), y=labels_array)
    assert(class_weights.size == num_classes)

    print("Class Weights: ", class_weights)
    return class_weights


def get_sampler_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    labels_array = dataset

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)

    sampler_weights = torch.zeros(len(labels_array))
    i = 0
    for label in labels_array:
        sampler_weights[i] = class_weights[int(label)]
        i += 1

    return sampler_weights


class SequentialLoss(nn.Module):
    def __init__(self):
        super(SequentialLoss, self).__init__()

    def forward(self, output, target, lengths):
        total_loss = 0
        for batch_idx in range(output.size(0)):
            weights = torch.arange(lengths[batch_idx]).float().cuda()/lengths[batch_idx].float()
            for sequence_idx in range(lengths[batch_idx]):
                out = output[batch_idx, sequence_idx, :].unsqueeze(0)
                tar = target[batch_idx].unsqueeze(0)
                total_loss += weights[sequence_idx] * F.cross_entropy(out, tar)
        return total_loss/output.size(0)


map_to_emo_family = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 1,
    6: 2,
    7: 1,
    8: 2,
    9: 3,
    10: 3,
    11: 3
}


def load_checkpoint(checkpoint_file):
    return torch.load(checkpoint_file)


def save_checkpoint(state, filename):
    filename = 'checkpoints/%s' % filename
    torch.save(state, filename)


class GroupCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GroupCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        output1 = output.clone()
        output1[:, 0] = output[:, 0]+output[:, 1]+output[:, 2]
        output1[:, 1] = output[:, 3]+output[:, 5]+output[:, 7]
        output1[:, 2] = output[:, 4]+output[:, 6]+output[:, 8]
        output1[:, 3] = output[:, 9]+output[:, 10]+output[:, 11]
        output1 = output[:, :4]
        return F.cross_entropy(output1, target)


def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=100):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def calc_gradients(params):
    grad_array = []
    _mean = []
    _max = []
    for param in params:
        grad_array.append(param.grad.data)
        _mean.append(torch.mean(param.grad.data))
        _max.append(torch.max(param.grad.data))
    print(np.mean(_mean))
    print(np.max(_max))


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def random_search():
    import random
    hidden_size = random.choice([50, 100, 150, 200, 250, 300])
    spatial_net_features = random.choice([50, 100, 150, 200, 250, 300])
    spatial_net_one_feature = random.choice([50, 100, 150, 200, 250, 300])
    num_input_lstm = random.choice([32, 64, 100, 128, 200, 256, 512])
    num_layers = random.choice([1, 2, 3, 4])
    bidirectional = random.choice([True, False])
    lr = random.choice([1e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3])
    step_size = random.choice([30, 60, 90, 120])
    epochs = random.choice([50, 100, 150, 200, 250, 300])
    weight_decay = random.choice([1e-4, 2e-4, 3e-4, 7e-4, 5e-4, 1e-3, 4e-3, 7e-3])
    dropout = random.choice([0, 0.2, 0.4, 0.5, 0.6, 0.8])
    batch_size = random.choice([16, 32, 64, 120])

    num_channels = random.choice([16, 32, 64, 128])
    kernel_size = random.choice([2, 3, 5, 7, 9, 11])
    num_tcn_layers = random.choice([2, 3, 4, 6, 8, 10])

    return {"hidden_size": hidden_size, "num_layers": num_layers, "bidirectional": bidirectional, "epochs": epochs,
            "step_size": step_size, "lr": lr,
            "weight_decay": weight_decay, "dropout": dropout, "batch_size": batch_size, "grad_clip": 0.1,
            "multiply_with_confidence": 0.3, "num_input_lstm": num_input_lstm,
            "num_channels": num_channels, "kernel_size": kernel_size, "num_tcn_layers": num_tcn_layers,
            "spatial_net_features": spatial_net_features,
            "spatial_net_one_feature": spatial_net_one_feature}
