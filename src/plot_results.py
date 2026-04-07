import matplotlib.pyplot as plt
import os
import re

def plot_accuracy(log_path, save_path):
    if not os.path.exists(log_path):
        print(f"Log not found at {log_path}")
        return

    rounds = []
    accuracies = []
    tasks = []

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Task: (\d+), Round: (\d+) Accuracy = ([\d.]+)%', line)
            if match:
                tasks.append(int(match.group(1)))
                rounds.append(int(match.group(2)))
                accuracies.append(float(match.group(3)))

    if not rounds:
        print("No accuracy data found in log.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', color='b', label='Global Accuracy')
    
    # Draw vertical lines for task changes
    last_task = tasks[0]
    for i, t in enumerate(tasks):
        if t != last_task:
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
            plt.text(i, max(accuracies)*0.9, f'Task {t}', color='r', rotation=90)
            last_task = t

    plt.title('Federated Learning Accuracy (GLFC - Tabular Dataset)')
    plt.xlabel('Global Round Index')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    log_file = './training_log/glfc/seed2021/log_tar_6.txt'
    save_file = '../results_accuracy.png'
    plot_accuracy(log_file, save_file)
