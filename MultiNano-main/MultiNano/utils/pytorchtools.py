import numpy as np
import torch
import os
class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model, optimizer):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer)
            self.counter = 0

    def recount(self):
        self.early_stop = False
        self.counter = 0

    def save_checkpoint(self, val_acc, model, optimizer):
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model ...')
        if torch.cuda.device_count() > 1:
            state = {'net': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
        else:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(self.path, "early_stopping_model_states.pt"))
        self.val_acc_max = val_acc
