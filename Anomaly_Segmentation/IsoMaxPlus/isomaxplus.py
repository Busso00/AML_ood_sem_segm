import torch.nn as nn
import torch.nn.functional as F
import torch


def input_reshape(x, n_class=20):
  x = x.permute(0,2,3,1)
  size = x.size()
  return x.reshape(-1,n_class), size

def output_reshape(x, size):
  x = x.reshape(size)
  return x.permute(0,3,1,2)

class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features=8*512*1024, num_classes=20, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        features, size = input_reshape(features, self.num_classes)
        distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features, dim=1), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        output = logits / self.temperature
        return output_reshape(output, size)


class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=True):
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        logits,_ = input_reshape(logits)
        #one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=20)
        #targets,_ = input_reshape(one_hot_targets)
        targets = targets.reshape(-1)
        distances = - logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        return loss
        