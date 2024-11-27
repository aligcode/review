import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomForwardBackward(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, weight, bias):
    # expected dimensions: input (8, 32) | weight (32, )
    print(f"in forward, input: {input.shape}, weight: {weight.shape}, bias: {bias.shape}")
    # addmm
    addmm = input @ weight.t() + bias
    ctx.save_for_backward(input, weight)
    return addmm

  @staticmethod
  def backward(ctx, grad_output): # grad of the loss wrt output
    print(f"in backward, grad_output: {grad_output.shape}, ctx: {ctx}")
    input, weight = ctx.saved_tensors
    # y = input * weight + bias
    # dl/d_input = dl/dy (grad_output) * dy/d_input (weight)
    grad_input = grad_output @ weight # last layer: [8, 16] <= [8, 3] @ [3, 16] | first layer: [8, 64] <= [8, 16] * [16, 64]
    # dl/d_weight = dl/dy (grad_output) * dy/dW (input)
    print(f"in backward, input_dims: {input.shape}")
    grad_weight = grad_output.t() @ input # last layer: [3 * 16] <= [8, 3] ? [8, 16]
    # dl/d_bias = dl/dy (grad_output) * dy/dB (0)
    grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias

class FullyConnectedNetworkWithCustomGrad(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_classes):
    super(FullyConnectedNetworkWithCustomGrad, self).__init__()

    # define two layers of parameters
    self.weights1 = nn.Parameter(torch.randn(hidden_dim, input_dim))
    self.bias1 = nn.Parameter(torch.randn(hidden_dim))
    self.weights2 = nn.Parameter(torch.randn(output_classes, hidden_dim))
    self.bias2 = nn.Parameter(torch.randn(output_classes))

    self.fc1 = lambda x: CustomForwardBackward.apply(x, self.weights1, self.bias1)
    self.fc2 = lambda x: CustomForwardBackward.apply(x, self.weights2, self.bias2)


  def forward(self, x: torch.tensor) -> torch.tensor:

    l1o = F.relu(self.fc1(x))
    preds = self.fc2(l1o)

    return preds

  def verify_jacobians(self):

    assert self.weights1.grad.shape == self.weights1.shape
    assert self.bias1.grad.shape == self.bias1.shape
    assert self.weights2.grad.shape == self.weights2.shape
    assert self.bias2.grad.shape == self.bias2.shape

  def print_gradients(self):
    if self.weights1.grad is None:
      return

    self.verify_jacobians()
    print(f"grad weights 1: {self.weights1.grad.shape}")
    print(f"grad bias 1: {self.bias1.grad.shape}")
    print(f"grad weights 2: {self.weights2.grad.shape}")
    print(f"grad bias 2: {self.bias2.grad.shape}")


class FullyConnectedNetwork(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_classes):
    super(FullyConnectedNetwork, self).__init__()

    # initial layer
    self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
    self.l2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
    self.l3 = nn.Linear(in_features=hidden_dim, out_features=output_classes, bias=True)


  def forward(self, x: torch.tensor) -> torch.tensor:
    l1o = F.relu(self.l1(x))
    l2o = F.relu(self.l2(l1o))
    pred_logits = self.l3(l2o)
    return pred_logits

if __name__ == '__main__':

  # model = FullyConnectedNetwork(input_dim=32, hidden_dim=16, output_classes=3)

  num_samples, input_dim = 8, 64
  # x = torch.rand(size=(num_samples, input_dim))
  # preds = model(x)
  # # print(f"Preds {preds}")

  print("Creating a fully connected model with custom gradients:")
  custom_grad_model = FullyConnectedNetworkWithCustomGrad(
      input_dim=64,
      hidden_dim=16,
      output_classes=3
  )

  x = torch.rand(size=(num_samples, input_dim))
  y = torch.rand(size=(num_samples, 3))
  preds = custom_grad_model(x)
  print(f"Custom grad model preds: {x}")

  loss_fn = nn.MSELoss()
  loss = loss_fn(preds, y)
  print(f"MSE Loss value: {loss}")

  print("gradient values before backward")
  custom_grad_model.print_gradients()
  print("Calling backward...")
  loss.backward()
  print("Gradients computed.")
  print("gradient values after backward")
  custom_grad_model.print_gradients()



