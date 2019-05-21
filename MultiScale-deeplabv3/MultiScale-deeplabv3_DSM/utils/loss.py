import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class DSMLosses(object):
    def __init__(self, batch_average=True, ignore_index=-1., cuda=False):
        self.ignore_index = ignore_index
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='logmse'):
        """Choices: ['logmse' or 'mse']"""
        if mode == 'mse':
            return self.mseLoss
        elif mode == 'logmse':
            return self.logmseLoss
        else:
            raise NotImplementedError

    def mseLoss(self, logit, target):
        logit = torch.squeeze(logit,1)

        '''
        criterion = nn.MSELoss(reduction='sum')
        if self.cuda:
            criterion = criterion.cuda()
        index = target==self.ignore_index
        logit[index] = 0
        target[index] = 0
        loss = criterion(logit, target)
        '''

        index = target==self.ignore_index
        num_valid = torch.sum(target[:] != self.ignore_index)
        logit[index] = 0
        target[index] = 0     

        if self.cuda:
            logit = logit.cuda()
            target = target.cuda()

        SE = (logit - target)*(logit - target)
        SE = torch.sum(SE[:])
        if self.batch_average:
            SE /= num_valid.float()

        return SE
    
    def logmseLoss(self, logit, target):
        logit = torch.squeeze(logit,1)
        
        index = target==self.ignore_index
        num_valid = torch.sum(target[:] != self.ignore_index)
        logit[index] = 0
        target[index] = 0

        if self.cuda:
            logit = logit.cuda()
            target = target.cuda()

        LogSE = torch.log((logit - target)*(logit - target)+1)
        LogSE = torch.sum(LogSE[:])
        if self.batch_average:
            LogSE /= num_valid.float()

        return LogSE

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




