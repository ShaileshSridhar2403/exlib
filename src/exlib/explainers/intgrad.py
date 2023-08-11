import torch
from tqdm import tqdm


def explain_classifier_with_intgrad(X, model,
                                    X0 = None,
                                    target_labels = None,
                                    num_steps = 100,
                                    progress_bar = False):
  """
  Explain a classification model with Integrated Gradients.
  """

  # Default baseline is zeros
  X0 = torch.zeros_like(X) if X0 is None else X0
  if X0 is None:
    X0 = torch.zeros_like(X)

  # Compute the target class if necessary
  if target_labels is None:
    with torch.no_grad():
      y = model(X)
    target_labels = y.argmax(dim=1)

  # The integrated gradients that we accumulate
  intgrad = torch.zeros_like(X)

  pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
  for k in pbar:
    ak = k / num_steps
    Xk = X0 + ak * (X - X0)
    Xk.requires_grad_()
    y = model(Xk)
    assert y.ndim == 2 # Because this is batched classification

    y_targets = y[:, target_labels]
    y_targets.sum().backward()

    intgrad += Xk.grad / num_steps # Recall that step_size = 1 / num_steps

  return {
      "intgrad" : intgrad
  }


