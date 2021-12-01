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
        tot = None
        for i in range(n):
            o1 = image_embedding[i].view(1, -1).repeat_interleave(n, dim=0)
            if tot == None:
                tot = self.cal_loss(o1, text_embedding, ground_truth[i])
            else:
                tot += self.cal_loss(o1, text_embedding, ground_truth[i])
            # print('tot=', tot)
        return tot
