model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/best_model.pth", map_location=DEVICE))
model.eval()

preds, truth = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        truth.append(yb.numpy())

preds = np.concatenate(preds)
truth = np.concatenate(truth)

cm = confusion_matrix(truth, preds)


plt.figure(figsize=(12,10))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

classes = le.classes_
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)

plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix_with_values.png", dpi=300)
plt.show()
