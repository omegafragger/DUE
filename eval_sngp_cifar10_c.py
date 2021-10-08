import argparse
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from due import dkl
from due.wide_resnet import WideResNet
from due.rff_laplace import Laplace
from due.resnet import resnet50, resnet110
from due.densenet import densenet121

from lib.datasets import get_dataset
from lib.evaluate_ood import get_ood_metrics, get_ood_metrics_cifar10_c
from lib.utils import get_results_directory, Hyperparameters, set_seed


parser = argparse.ArgumentParser()

architecture_dict = {
    "WRN": 'wrn',
    "ResNet50": "resnet50",
    "ResNet110": "resnet110",
    "DenseNet121": "densenet121"
}

parser.add_argument(
    "--model",
    default="WRN",
    choices=["WRN", "ResNet50", "ResNet110", "DenseNet121"],
    help="Pick an architecture"
)

parser.add_argument(
    "--data_path",
    default='./',
    type=str,
    help='Path to CIFAR-10-C'
)

parser.add_argument(
    "--corruption_type",
    type=str,
    default="brightness",
    help='Corruption Type'
)

parser.add_argument(
    "--corruption_intensity",
    type=int,
    default=1,
    help="Corruption Intensity"
)

args = parser.parse_args()

# _, _, _, cifar10_test_set = get_dataset('CIFAR10')
# _, _, _, cifar10_c_test_set = get_dataset('CIFAR10_C',
#                                           root=args.data_path,
#                                           corruption_type=args.corruption_type,
#                                           corruption_intensity=args.corruption_intensity)


models = []

for i in range(5):
    if args.model == "WRN":
        feature_extractor = WideResNet(
            32,
            True,
            True,
            dropout_rate=0.3,
            coeff=3,
            n_power_iterations=1,
        )
    elif args.model == "ResNet50":
        feature_extractor = resnet50(
            spectral_normalization=True,
            mod=False
        )
    elif args.model == "ResNet110":
        feature_extractor = resnet110(
            spectral_normalization=True,
            mod=False
        )
    elif args.model == "DenseNet121":
        feature_extractor = densenet121(
            spectral_normalization=True,
            mod=False
        )

    # Defaults from SNGP in uncertainty-baselines
    if args.model == "WRN":
        num_deep_features = 640
    elif args.model == "ResNet50" or args.model == "ResNet110":
        num_deep_features = 2048
    elif args.model == "DenseNet121":
        num_deep_features = 1024
    num_gp_features = 128
    num_random_features = 1024
    normalize_gp_features = True
    lengthscale = 2
    mean_field_factor = 25
    num_data = 50000
    num_classes = 10

    model = Laplace(
        feature_extractor,
        num_deep_features,
        num_gp_features,
        normalize_gp_features,
        num_random_features,
        num_classes,
        num_data,
        500,
        mean_field_factor,
        lengthscale,
    )

    model = model.cuda()
    model.load_state_dict(torch.load(f'./runs/cifar10/{architecture_dict[args.model]}/Run{i+1}/model.pt'))
    models.append(model)


aurocs = []

for i in range(5):
    _, _, auroc_cifar10_c, _ = get_ood_metrics_cifar10_c("CIFAR10",
                                                         "CIFAR10_C",
                                                         models[i],
                                                         root=args.data_path,
                                                         corruption_type=args.corruption_type,
                                                         corruption_intensity=args.corruption_intensity)

    aurocs.append(auroc_cifar10_c)


aurocs = torch.tensor(aurocs)
mean_auroc = torch.mean(aurocs).item()
std_auroc = torch.std(aurocs).item() / math.sqrt(aurocs.shape[0])

res_dict = {
    'mean_auroc': mean_auroc,
    'std_auroc': std_auroc
}

with open(f'res_{args.model}_{args.corruption_type}_{args.corruption_intensity}.json', 'w+') as fp:
    json.dump(res_dict, fp)



