import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def add_self_loops(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device).unsqueeze(1)
    loop_index = torch.cat([loop_index, loop_index], dim=1).t()
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, None

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def reconstruction_loss(input_data, reconstructed_data, batch_indices):
    reconstructed_data_per_atom = reconstructed_data[batch_indices]
    error_tensor = input_data - reconstructed_data_per_atom
    squared_error_tensor = torch.pow(error_tensor, 2)
    mean_squared_error = torch.mean(squared_error_tensor)
    return mean_squared_error

def train(model, data_loader, optimizer, prediction_loss_function, reconstruction_loss_function, alpha_value=0.5):
    model.train()
    cumulative_loss = 0.0
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        prediction_output, reconstruction_output = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        prediction_loss_value = prediction_loss_function(prediction_output, batch_data.y)
        reconstruction_loss_value = reconstruction_loss(batch_data.x, reconstruction_output, batch_data.batch)
        weighted_prediction_loss = alpha_value * prediction_loss_value
        weighted_reconstruction_loss = (1.0 - alpha_value) * reconstruction_loss_value
        combined_loss_value = weighted_prediction_loss + weighted_reconstruction_loss
        combined_loss_value.backward()
        optimizer.step()
        cumulative_loss += combined_loss_value.item()
    total_batches = len(data_loader)
    average_loss_value = cumulative_loss / total_batches
    return average_loss_value

def evaluate(model, data_loader):
    model.eval()
    actual_labels = []
    predicted_probabilities = []
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            model_output, _ = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            actual_labels_cpu = batch_data.y.cpu().numpy().tolist()
            predicted_probabilities_cpu = model_output.cpu().numpy().tolist()
            actual_labels.extend(actual_labels_cpu)
            predicted_probabilities.extend(predicted_probabilities_cpu)

    numpy_actual_labels = np.array(actual_labels)
    numpy_predicted_probabilities = np.array(predicted_probabilities)

    binary_predicted_labels = (numpy_predicted_probabilities >= 0.5).astype(int)
    binary_actual_labels = numpy_actual_labels.astype(int)

    accuracy = accuracy_score(binary_actual_labels, binary_predicted_labels)
    balanced_accuracy = balanced_accuracy_score(binary_actual_labels, binary_predicted_labels)
    matthews_correlation_coefficient = matthews_corrcoef(binary_actual_labels, binary_predicted_labels)
    try:
        area_under_roc_curve = roc_auc_score(binary_actual_labels, numpy_predicted_probabilities)
    except ValueError:
        area_under_roc_curve = 0.0
    precision = precision_score(binary_actual_labels, binary_predicted_labels, zero_division=1)
    recall = recall_score(binary_actual_labels, binary_predicted_labels)
    f1_score_value = f1_score(binary_actual_labels, binary_predicted_labels)

    confusion_matrix_result = confusion_matrix(binary_actual_labels, binary_predicted_labels)
    true_negative_count, false_positive_count, false_negative_count, true_positive_count = confusion_matrix_result.ravel()
    sum_true_positives_false_negatives = true_positive_count + false_negative_count
    sum_true_negatives_false_positives = true_negative_count + false_positive_count
    sensitivity = true_positive_count / sum_true_positives_false_negatives if sum_true_positives_false_negatives > 0 else 0.0
    specificity = true_negative_count / sum_true_negatives_false_positives if sum_true_negatives_false_positives > 0 else 0.0

    return accuracy, balanced_accuracy, matthews_correlation_coefficient, area_under_roc_curve, precision, recall, f1_score_value, sensitivity, specificity