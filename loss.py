import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch


class NTXentLoss(nn.Module):

    def __init__(self):
        super(NTXentLoss, self).__init__()
        
    def forward(self, p1, p2):  
        
        # cosine(a, b) = a . b / ||a|| ||b||
        batch_size = p1.shape[0]
        p_cat = torch.cat([p1, p2], dim=0) # 2*batch, 32
        
        # dot 
        p_cat_dot = p_cat @ p_cat.T  # 2*batch, 32  32, 2*batch => [2*batch, 2*batch]
        
        # norms
        p_cat_norm = torch.norm(p_cat, p=2, dim=-1, keepdim=True) # 2*batch, 1
        p_cat_norm_prod = torch.mm(p_cat_norm, p_cat_norm.T) # 2*batch, 2*batch
         
        # cosine similarity [2*batch, 2*batch]
        p_cosine = torch.divide(p_cat_dot, p_cat_norm_prod + 1e-8)
        assert p_cosine.shape == (2*batch_size, 2*batch_size)

        # p_cosine: [2*batch_size, 2*batch_size]: (i, j) [:batch_size, batch_size:] (j, i) [batch_size:, :batch_size]
        i_j_pos = torch.diag(p_cosine[:batch_size, batch_size:]) # batch_size
        j_i_pos = torch.diag(p_cosine[batch_size:, :batch_size]) # batch_size
        
        assert i_j_pos.shape[0] == batch_size
        assert j_i_pos.shape[0] == batch_size
        
        p_cosine_exp = torch.exp(p_cosine)
        denominator = torch.sum(p_cosine_exp, dim=1) # [2*batch_size]
    
        # create a mask to get non corrsponding (i, k) from p_cosine where i != k        
        ij_numerator = torch.exp(i_j_pos) # [batch_size]
        ji_numerator = torch.exp(j_i_pos) # batch_size
        
        ijloss = -torch.log(ij_numerator/denominator[:batch_size])
        jiloss = -torch.log(ji_numerator/denominator[batch_size:])
        
        total_loss = (ijloss + jiloss).mean()
        
        return total_loss
    
class BootstrappedCrossEntropy(nn.Module):
    # weight=None, ignore_index=-100, reduction='mean', bootstrap=bootstrap)
    
    def __init__(self, weight, ignore_index, reduction, bootstrap, beta):
        super(BootstrappedCrossEntropy, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.bootstrap = bootstrap
        print(f"Using bootstrapping loss on mode: {bootstrap}")
        self.ce_loss = CrossEntropyLoss(
            weight=self.weight, 
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        self.beta = beta
        
    
    def soft_bootstrap_ce_loss(self, pred, label):
        label_one_hot = F.one_hot(label.long(), num_classes=pred.size(1))
        pred_probs = F.softmax(pred, dim=-1)
        label_bootstrapped = self.beta * label_one_hot + (1.0 - self.beta) * pred_probs
        log_probs = F.log_softmax(pred, dim=-1)
        loss = - torch.sum(label_bootstrapped * log_probs, dim=-1).mean()
        return loss
    
    
    def hard_bootstrap_ce_loss(self, pred, label):
        label_one_hot = F.one_hot(label.long(), num_classes=pred.size(1))
        pred_argmax = torch.argmax(F.softmax(pred, dim=-1), dim=-1)
        pred_one_hot = F.one_hot(pred_argmax, num_classes=pred.size(1))
        
        label_bootstrapped = self.beta * label_one_hot + (1.0 - self.beta) * pred_one_hot
        loss_probs = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(label_bootstrapped * loss_probs, dim=-1).mean()
        return loss 
        
    def forward(self, pred, label):
        if self.bootstrap == 'no_bootstrap':
            return self.ce_loss(pred, label)
        
        elif self.bootstrap == 'soft':
            return self.soft_bootstrap_ce_loss(pred, label)
        
        elif self.bootstrap == 'hard':
            return self.hard_bootstrap_ce_loss(pred, label)
        
        else:
            raise NotImplementedError(f"bootstrap type {self.bootstrap} is not implemented.")
        

if __name__ == '__main__':
    
    batch_size = 8
    hidden_dim = 256
    p1 = torch.randn(batch_size, hidden_dim)
    p2 = torch.randn(batch_size, hidden_dim)
    
    simclr_loss = NTXentLoss()
    loss = simclr_loss(p1, p2).item()
    print(f"SimCLR loss value is {loss}")