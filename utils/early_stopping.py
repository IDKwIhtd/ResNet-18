import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path="checkpoint.pth"):
        """
        조기 종료 클래스
        Args:
            patience (int): 개선되지 않아도 참을 epoch 수
            verbose (bool): 개선 시 출력 여부
            delta (float): 개선으로 간주할 최소 변화량
            path (str): 모델 저장 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        """최고 성능 모델 저장"""
        if self.verbose:
            print(f"Validation loss improved. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path),
        self.val_loss_min = val_loss
