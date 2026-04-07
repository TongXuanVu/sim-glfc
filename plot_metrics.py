import re
import matplotlib.pyplot as plt
import os

log_path = r'c:\Users\Admin\Desktop\glfc\sim-glfc\src\training_log\glfc\seed2021\log_tar_6.txt'
output_plot = r'c:\Users\Admin\Desktop\glfc\sim-glfc\training_metrics.png'

rounds = []
train_losses = []
eval_losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
tasks = []

if not os.path.exists(log_path):
    print(f"Log file not found at {log_path}")
    exit(1)

with open(log_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line or not line.startswith("Task:"): continue
    
    # Extract Task and Round
    match = re.search(r"Task: (\d+), Round: (\d+)", line)
    if match:
        tasks.append(int(match.group(1)))
        rounds.append(int(match.group(2)))
    
    # Extract TrainLoss
    match = re.search(r"TrainLoss: ([\d.e-]+)", line)
    if match: train_losses.append(float(match.group(1)))
    
    # Extract EvalLoss
    match = re.search(r"EvalLoss: ([\d.e-]+)", line)
    if match: eval_losses.append(float(match.group(1)))
    
    # Extract Acc
    match = re.search(r"Acc: ([\d.]+)%", line)
    if match: accuracies.append(float(match.group(1)))
    
    # Extract Prec
    match = re.search(r"Prec: ([\d.]+)%", line)
    if match: precisions.append(float(match.group(1)))

    # Extract Rec
    match = re.search(r"Rec: ([\d.]+)%", line)
    if match: recalls.append(float(match.group(1)))

    # Extract F1
    match = re.search(r"F1: ([\d.]+)%", line)
    if match: f1_scores.append(float(match.group(1)))

if not rounds:
    print("No data parsed from log file.")
    exit(1)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.3)

# Plot 1: Accuracy & F1
ax1.plot(rounds, accuracies, label='Accuracy (%)', marker='o', color='blue', linewidth=2)
ax1.plot(rounds, f1_scores, label='F1-Score (%)', marker='x', color='green', linestyle='--')
ax1.set_title('Training Progress: Accuracy & F1-Score', fontsize=14, fontweight='bold')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# Highlight Task transitions
task_changes = [0]
for i in range(1, len(rounds)):
    if tasks[i] != tasks[i-1]:
        task_changes.append(rounds[i])
        ax1.axvline(x=rounds[i], color='gray', linestyle='--', alpha=0.5)
        ax1.text(rounds[i], max(accuracies), f'Task {tasks[i]}', rotation=90, verticalalignment='bottom')

# Plot 2: Loss
ax2.plot(rounds, train_losses, label='Train Loss', marker='.', color='red')
ax2.plot(rounds, eval_losses, label='Eval Loss (Cross-Entropy)', marker='.', color='orange')
ax2.set_title('Training Progress: Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Round', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.set_yscale('log') # Loss often looks better in log scale if ranges differ wildly
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()
for tc in task_changes:
    ax2.axvline(x=tc, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Precision & Recall
ax3.plot(rounds, precisions, label='Precision (%)', marker='s', color='#9b59b6')
ax3.plot(rounds, recalls, label='Recall (%)', marker='^', color='#e67e22')
ax3.set_title('Training Progress: Precision & Recall', fontsize=14, fontweight='bold')
ax3.set_xlabel('Round', fontsize=12)
ax3.set_ylabel('Percentage (%)', fontsize=12)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend()
for tc in task_changes:
    ax3.axvline(x=tc, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(output_plot, dpi=150)
print(f"Plot saved successfully to {output_plot}")
