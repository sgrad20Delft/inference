import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
os.makedirs("graph", exist_ok=True)

# === Load CSV files ===
# Latency
efficientnet_latency_darwin = pd.read_csv("latency_performance_mode_efficientnet.csv")
vit_latency_darwin = pd.read_csv("latency_performance_mode_huggingface.csv")
efficientnet_latency_win = pd.read_csv("latency_performance_mode_efficientNet_windows.csv")
vit_latency_win = pd.read_csv("latency_performance_mode_vit_windows.csv")

# Results
efficientnet_results_darwin = pd.read_csv("results_efficientnet_b0_darwin.csv")
vit_results_darwin = pd.read_csv("results_vit-base-patch16-224_darwin.csv")
efficientnet_results_win = pd.read_csv("results_efficientnet_b0_windows.csv")
vit_results_win = pd.read_csv("results_vit-base-patch16-224_windows.csv")

# === Annotate OS and model ===
def label_latency(df, model, os_name):
    df["model"] = model
    df["os"] = os_name
    return df

latency_darwin = pd.concat([
    label_latency(efficientnet_latency_darwin, "efficientnet_b0", "darwin"),
    label_latency(vit_latency_darwin, "vit_base_patch16_224", "darwin")
])
latency_windows = pd.concat([
    label_latency(efficientnet_latency_win, "efficientnet_b0", "windows"),
    label_latency(vit_latency_win, "vit_base_patch16_224", "windows")
])
latency_all = pd.concat([latency_darwin, latency_windows], ignore_index=True)

def label_results(df, os_name):
    df["os"] = os_name
    df["evaluated_samples"] = 7404
    df["energy_per_inference_wh"] = df["energy_wh"] / df["evaluated_samples"]
    return df

results_darwin = pd.concat([
    label_results(efficientnet_results_darwin, "darwin"),
    label_results(vit_results_darwin, "darwin")
])
results_windows = pd.concat([
    label_results(efficientnet_results_win, "windows"),
    label_results(vit_results_win, "windows")
])
results_all = pd.concat([results_darwin, results_windows], ignore_index=True)

# === Combined Plot 1: Energy per Inference ===
plt.figure(figsize=(10, 6))
sns.barplot(data=results_all, x="model_architecture", y="energy_per_inference_wh", hue="os", palette="pastel")
plt.title("Energy per Inference (Wh) — Combined")
plt.ylabel("Energy per Inference (Wh)")
plt.xlabel("Model Architecture")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph/energy_per_inference_combined.png")
plt.close()

# === Combined Plot 2: Latency per Inference ===
latency_summary_all = latency_all.groupby(["os", "model"])["latency_ms"].agg(["mean", "std"]).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=latency_summary_all, x="model", y="mean", hue="os", palette="Set2", capsize=0.2)
plt.title("Latency per Inference (ms) — Combined")
plt.ylabel("Latency (ms)")
plt.xlabel("Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph/latency_per_inference_combined.png")
plt.close()

# === Combined Plot 3: Normalized Energy Score ===
plt.figure(figsize=(10, 6))
sns.barplot(data=results_all, x="model_architecture", y="normalized_energy", hue="os", palette="muted")
plt.title("Normalized Energy Score — Combined")
plt.ylabel("Normalized Energy")
plt.xlabel("Model Architecture")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph/normalized_energy_score_combined.png")
plt.close()

# === Accuracy vs Energy (Separate per OS) ===
def plot_accuracy_vs_energy(df, os_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="energy_wh", y="accuracy", hue="model_architecture", s=100)
    plt.title(f"Accuracy vs Energy — {os_name}")
    plt.xlabel("Total Energy (Wh)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graph/accuracy_vs_energy_{os_name}.png")
    plt.close()

plot_accuracy_vs_energy(results_darwin, "darwin")
plot_accuracy_vs_energy(results_windows, "windows")

print("All graphs saved under the 'graph/' directory.")
