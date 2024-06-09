import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the architecture data for each model
models = {
    "Model 1 GCN": [
        ("GCN Layer", 32),
        ("GCN Layer", 64),
        ("ConCat (Jump)", 96),
        ("Global Pool", 96),
        ("Fully Connected", 128),
        ("Output", 1)
    ],
    "Model 2 GAT": [
        ("GAT Layer (4 heads)", 64),
        ("GAT Layer (8 heads)", 256),
        ("ConCat (Jump)", 320),
        ("Global Pool", 320),
        ("Fully Connected", 128),
        ("Output", 1)
    ],
    "Model 3 GIN": [
        ("GIN Layer", 32),
        ("GIN Layer", 32),
        ("Global Pool", 32),
        ("Fully Connected", 32),
        ("Output", 1)
    ],
    "Model 4 GCN+GAT-S": [
        ("GCN Layer", 16),
        ("GAT Layer (8 heads)", 128),
        ("ConCat (Jump)", 144),
        ("Global Pool", 144),
        ("Fully Connected", 128),
        ("Output", 1)
    ],
    "Model 5 GCN+GAT-D": [
        ("GCN Layer", 16),
        ("GCN Layer", 32),
        ("GAT Layer (4 heads)", 128),
        ("GAT Layer (8 heads)", 256),
        ("ConCat (Jump)", 432),
        ("Global Pool", 432),
        ("Fully Connected", 128),
        ("Output", 1)
    ]
}

# reverse the layers for bottom-to-top plotting
for model_name, layers in models.items():
    models[model_name] = list(reversed(layers))

# Function to plot the architecture
def plot_architecture(model_name, layers):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title(model_name)
    ax.axis('off')

    # Variables to adjust layout and spacing
    x, y = 0, 0
    width, height = 1.2, 3.5
    
    
    gap = 0.8

    # Reverse the layers for bottom-to-top plotting
    layers_reversed = list(reversed(layers))

    for i, (layer, dim) in enumerate(layers_reversed):
        # Draw rectangles
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='royalblue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, f"{layer}\n({dim}D)", ha='center', va='center', fontsize=8)

        # Draw arrows if not the last layer
        # if i == 0:
        #     arrow = patches.FancyArrow(x + width / 2, y - gap - gap, 0, gap, width=0.01, head_width=0.1, head_length=0.05, length_includes_head=True, color='black')
        #     ax.add_patch(arrow)
        
        y += height + gap

    ax.set_xlim(-1, x + width + 1)
    ax.set_ylim(-1, y)
    # plt.show()
    plt.savefig(f"{model_name}.png", dpi=300)

# Plot the architectures
for model_name, layers in models.items():
    plot_architecture(model_name, layers)
