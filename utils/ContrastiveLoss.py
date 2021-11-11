import torch.nn
import torch.nn.functional as F


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def cal_loss(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    def forward(self, image_embedding, text_embedding, ground_truth):
        n = len(image_embedding)
        output1 = []
        output2 = []
        label = []
        for i in range(n):
            for j in range(n):
                output1.append(list(image_embedding[i]))
                output2.append(list(text_embedding[j]))
                label.append(ground_truth[i][j])
        output1 = torch.tensor(output1)
        output2 = torch.tensor(output2)
        label = torch.tensor(label)
        return self.cal_loss(output1, output2, label)
