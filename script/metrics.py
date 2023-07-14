"""
Process model predictions and calculate metrics with ground truths
"""

import sklearn.metrics
import torch
import numpy as np


class ClassificationMetrics:

    def __init__(self, truths, outputs):
        """
        N is number of samples and C is number of classes
        :param truths: (N)
        :param outputs: (N) or (N, C) or (C, N)
        """

        # process and verify inputs
        if isinstance(truths, torch.Tensor):
            truths = truths.detach().cpu()
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu()
        truths_arr = np.array(truths).squeeze()
        outputs_arr = np.array(outputs).squeeze()
        truths_shape = truths_arr.shape
        outputs_shape = outputs_arr.shape

        self.truths = truths_arr  # ground truth predictions
        self._outputs = None  # scores for each class if outputs is 2d
        self.preds = None  # predictions with the largest score

        if len(truths_shape) != 1:
            raise Exception(f"Expected truths to be 1-d array, but got {truths_shape}")
        if len(outputs_shape) not in (1, 2):
            raise Exception(f"Expected outputs to be 1-d/2-d array, but got {outputs_shape}")

        if len(outputs_shape) == 1:
            if truths_shape[0] == outputs_shape[0]:
                self.preds = outputs_arr
            else:
                e = f"Expected outputs to have same length with truths {truths_shape}, but got {outputs_shape}"
                raise Exception(e)
        elif len(outputs_shape) == 2:
            self._outputs = outputs_arr
            if truths_shape[0] == outputs_shape[0]:
                self.preds = np.argmax(outputs_arr, axis=1)
            elif truths_shape[0] == outputs_shape[1]:
                self.preds = np.argmax(outputs_arr, axis=0)
                self._outputs = np.transpose(self._outputs)
            else:
                e = f"Expected outputs to have shape (-1, {truths_shape[0]}) or ({truths_shape[0]}, -1), " \
                    f"but got {outputs_shape}"
                raise Exception(e)

        assert self.truths.shape == self.preds.shape

    @property
    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.truths, self.preds)

    @property
    def precision(self):
        return sklearn.metrics.precision_score(self.truths, self.preds, average="macro", zero_division=0)

    @property
    def recall(self):
        return sklearn.metrics.recall_score(self.truths, self.preds, average="macro", zero_division=0)

    @property
    def f1_score(self):
        return sklearn.metrics.f1_score(self.truths, self.preds, average="macro", zero_division=0)

    def topK_accuracy(self, k=5):
        if self._outputs is None:
            raise Exception("Top K accuracy not supported with 1d outputs")
        return sklearn.metrics.top_k_accuracy_score(self.truths, self._outputs, k=k)

    def print_report(self):
        if self._outputs is None:
            print(f"Accuracy: {self.accuracy:2.2%} | Precision: {self.precision:.4f}")
            print(f"Recall:   {self.recall:.4f} | F1 score:  {self.f1_score:.4f}")
        else:
            print(f"Top 1 Accuracy: {self.accuracy:2.2%} | Top 5 Accuracy: {self.topK_accuracy(k=5):2.2%}")
            print(f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1 score: {self.f1_score:.4f}")


def test():
    # truths = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).to("cuda")
    # preds = torch.tensor([0, 1, 1, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 2]).to("cuda")
    # truths = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    # outputs = np.array([0, 1, 1, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 2])
    truths = [0, 1, 2, 0, 1]
    # outputs = [0, 1, 1, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 2]
    outputs = [[0, 1, 1], [2, 1, 0], [0, 1, 2], [0, 1, 0], [0, 2, 2]]

    # print(np.array(preds.cpu()))
    metrics = ClassificationMetrics(truths, outputs)
    print(metrics.preds)
    metrics.print_report()


if __name__ == '__main__':
    test()