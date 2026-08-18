import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. DATASET: Synthetic Shapes (Triangles, Circles, Squares, Stars)
# =====================================================================
class SyntheticShapesDataset(Dataset):
    """
    Generates 28x28 grayscale images of 4 classes:
    0: Square, 1: Circle, 2: Triangle, 3: Line
    """
    def __init__(self, num_samples_per_class=100):
        self.images = []
        self.labels = []
        
        for cls in range(4):
            for _ in range(num_samples_per_class):
                img = np.zeros((28, 28), dtype=np.float32)
                
                if cls == 0:  # Square
                    img[6:22, 6:22] = 1.0
                elif cls == 1:  # Circle
                    y, x = np.ogrid[:28, :28]
                    mask = (x - 14)**2 + (y - 14)**2 <= 8**2
                    img[mask] = 1.0
                elif cls == 2:  # Triangle
                    for r in range(6, 22):
                        width = (r - 6) // 2
                        img[r, 14 - width : 14 + width + 1] = 1.0
                elif cls == 3:  # Horizontal Line
                    img[13:16, 4:24] = 1.0
                
                # Add random noise to make tasks unique
                img += np.random.normal(0, 0.1, img.shape)
                img = np.clip(img, 0, 1)
                
                self.images.append(img[None, :, :])  # Shape: (1, 28, 28)
                self.labels.append(cls)
                
        self.images = torch.tensor(np.array(self.images), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# =====================================================================
# 2. TASK DATALOADER: Generates Few-Shot "Support" and "Query" Tasks
# =====================================================================
class MetaTaskBatchSampler:
    """
    Samples a meta-batch of classification tasks.
    Each task randomly picks 'n_ways' classes, taking 'k_shots' for support
    and 'q_queries' for evaluation.
    """
    def __init__(self, dataset, n_ways=2, k_shots=3, q_queries=3, batch_size=4):
        self.dataset = dataset
        self.n_ways = n_ways        # Number of classes per task
        self.k_shots = k_shots      # Examples per class for Inner Loop (Support)
        self.q_queries = q_queries  # Examples per class for Outer Loop (Query)
        self.batch_size = batch_size # Number of tasks per meta-step
        
        # Group indices by class label
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            l = label.item()
            if l not in self.class_indices:
                self.class_indices[l] = []
            self.class_indices[l].append(idx)

    def sample_task(self):
        # Pick n_ways distinct classes
        chosen_classes = np.random.choice(list(self.class_indices.keys()), self.n_ways, replace=False)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for new_label, cls in enumerate(chosen_classes):
            # Sample (k_shots + q_queries) images for this class
            sampled_indices = np.random.choice(
                self.class_indices[cls], 
                self.k_shots + self.q_queries, 
                replace=False
            )
            
            supp_idxs = sampled_indices[:self.k_shots]
            query_idxs = sampled_indices[self.k_shots:]
            
            for idx in supp_idxs:
                support_x.append(self.dataset.images[idx])
                support_y.append(new_label)  # Map to 0..n_ways-1
                
            for idx in query_idxs:
                query_x.append(self.dataset.images[idx])
                query_y.append(new_label)

        return (
            torch.stack(support_x), torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x), torch.tensor(query_y, dtype=torch.long)
        )


# =====================================================================
# 3. MODEL: Functional ConvNet
# =====================================================================

class FunctionalConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Define layers directly as class attributes to guarantee exact names
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.fc    = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x, params=None):
        if params is None:
            # Standard forward pass
            x = torch.relu(self.bn1(self.conv1(x)))
            x = nn.functional.max_pool2d(x, 2)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = nn.functional.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        
        # Functional forward pass using exact parameter keys
        x = nn.functional.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1)
        x = nn.functional.batch_norm(
            x, running_mean=None, running_var=None,
            weight=params['bn1.weight'], bias=params['bn1.bias'], training=True
        )
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        x = nn.functional.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=1)
        x = nn.functional.batch_norm(
            x, running_mean=None, running_var=None,
            weight=params['bn2.weight'], bias=params['bn2.bias'], training=True
        )
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = nn.functional.linear(x, params['fc.weight'], params['fc.bias'])
        return x


# =====================================================================
# 4. META-TRAINING LOOP
# =====================================================================
dataset = SyntheticShapesDataset(num_samples_per_class=100000)
task_sampler = MetaTaskBatchSampler(dataset, n_ways=2, k_shots=3, q_queries=3, batch_size=4)

model = FunctionalConvNet(num_classes=2)
meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
inner_lr = 0.05
criterion = nn.CrossEntropyLoss()

print("--- Starting Meta-Training ---")
for meta_epoch in range(1, 501):
    meta_optimizer.zero_grad()
    meta_loss = 0.0
    
    # Process a batch of tasks (e.g., 4 tasks per meta-step)
    for _ in range(task_sampler.batch_size):
        supp_x, supp_y, query_x, query_y = task_sampler.sample_task()
        
        # A. Inner Loop: Copy theta_meta parameters
        fast_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        # Take 1 gradient step on the Support Set
        supp_logits = model(supp_x, params=fast_weights)
        inner_loss = criterion(supp_logits, supp_y)
        
        grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
        fast_weights = {
            name: param - inner_lr * g 
            for (name, param), g in zip(fast_weights.items(), grads)
        }
        
        # B. Outer Loop: Evaluate adapted parameters on Query Set
        query_logits = model(query_x, params=fast_weights)
        q_loss = criterion(query_logits, query_y)
        meta_loss += q_loss / task_sampler.batch_size

    # Outer Step: Update theta_meta
    meta_loss.backward()
    meta_optimizer.step()

    if meta_epoch % 10 == 0:
        print(f"Meta-Epoch [{meta_epoch}/50] - Outer Loss: {meta_loss.item():.4f}")


# =====================================================================
# 5. NEW TASK ADAPTATION (TEST TIME INFERENCE)
# =====================================================================
print("\n--- Meta-Training Finished ---")
print("Testing on a Brand-New Unseen Task (Circles vs. Triangles)...")

# Sample 1 new task with 3 support images per class
test_supp_x, test_supp_y, test_query_x, test_query_y = task_sampler.sample_task()

# Step 1: Copy base weights theta_meta
test_adapted_weights = {name: param.clone() for name, param in model.named_parameters()}

# Step 2: Adapt to the 3 support examples in 3 fast gradient steps
for step in range(1, 4):
    logits = model(test_supp_x, params=test_adapted_weights)
    loss = criterion(logits, test_supp_y)
    
    grads = torch.autograd.grad(loss, test_adapted_weights.values())
    test_adapted_weights = {
        name: param - inner_lr * g 
        for (name, param), g in zip(test_adapted_weights.items(), grads)
    }

# Step 3: Classify Query Set using the adapted parameters
with torch.no_grad():
    final_query_logits = model(test_query_x, params=test_adapted_weights)
    preds = torch.argmax(final_query_logits, dim=1)
    acc = (preds == test_query_y).float().mean() * 100

print(f"Predictions on Query Set: {preds.tolist()}")
print(f"Ground Truth Labels:    {test_query_y.tolist()}")
print(f"Adapted Model Accuracy: {acc.item():.1f}%")