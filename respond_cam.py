'''
Author: Guanan Zhao
'''
import torch
import torch.nn.functional as F

def get_activation_hook(layer_name):
    activations = {}
    def hook(model, input, output):
        activations[layer_name] = output.detach()
    return hook, activations

def grad_cam(cnn_model, data, class_index, target_layer_name):
    return x_cam(cnn_model, data, class_index, target_layer_name, respond=False)

def respond_cam(cnn_model, data, class_index, target_layer_name):
    return x_cam(cnn_model, data, class_index, target_layer_name, respond=True)

def x_cam(cnn_model, data, class_index, target_layer_name, respond):
    # Set the model in evaluation mode
    cnn_model.eval()

    # Register hook for activations
    hook, activations = get_activation_hook(target_layer_name)
    getattr(cnn_model, target_layer_name).register_forward_hook(hook)

    # Forward pass
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # add batch dimension
    output = cnn_model(data_tensor)
    class_score = output[0, class_index]

    # Backward pass for gradients
    cnn_model.zero_grad()
    class_score.backward()

    # Get activations and gradients
    activation = activations[target_layer_name].squeeze()
    gradient = next(cnn_model.parameters()).grad  # assuming the first parameter has the required grad

    # Weight calculation
    if respond:
        weights = torch.sum(activation * gradient, dim=(1, 2)) / (torch.sum(activation, dim=(1, 2)) + 1e-10)
    else:
        weights = torch.mean(gradient, dim=(1, 2))

    # CAM generation
    cam = torch.sum(activation * weights.view(1, -1, 1, 1), dim=1)
    return cam


def get_activation_and_gradient_hook(layer_name):
    activations = {}
    gradients = {}

    def forward_hook(model, input, output):
        activations[layer_name] = output.detach()

    def backward_hook(model, grad_input, grad_output):
        gradients[layer_name] = grad_output[0].detach()

    return forward_hook, backward_hook, activations, gradients

def get_all_scores_and_camsums(cnn_model, target_layer_name, dataset):
    cnn_model.eval()

    # Register hooks for activations and gradients
    forward_hook, backward_hook, activations, gradients = get_activation_and_gradient_hook(target_layer_name)
    layer = getattr(cnn_model, target_layer_name)
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    camsums_grad = []
    camsums_respond = []
    all_scores = []

    for data in dataset:
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        output = cnn_model(data_tensor)

        class_count = output.shape[1]
        camsum_grad = []
        camsum_respond = []
        scores = []

        for class_index in range(class_count):
            cnn_model.zero_grad()
            score = output[0, class_index]
            score.backward(retain_graph=True)

            activation = activations[target_layer_name].squeeze()
            gradient = gradients[target_layer_name].squeeze()

            grad_weights = torch.mean(gradient, dim=(1, 2))
            grad_cam = torch.sum(activation * grad_weights.view(1, -1, 1, 1), dim=1)
            camsum_grad.append(torch.sum(grad_cam).item())

            respond_weights = torch.sum(activation * gradient, dim=(1, 2)) / (torch.sum(activation, dim=(1, 2)) + 1e-10)
            respond_cam = torch.sum(activation * respond_weights.view(1, -1, 1, 1), dim=1)
            camsum_respond.append(torch.sum(respond_cam).item())

            scores.append(score.item())

        camsums_grad.append(np.array(camsum_grad))
        camsums_respond.append(np.array(camsum_respond))
        all_scores.append(np.array(scores))

    return camsums_grad, camsums_respond, all_scores

