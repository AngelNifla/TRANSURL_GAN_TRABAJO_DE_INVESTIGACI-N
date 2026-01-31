import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from data_processing import dataPreprocess_bert, spiltDatast_bert
from Model_MMA import Model


############################################
# TRAIN FUNCTION — returns epoch loss
############################################
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    batches = 0

    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        x1, x2, x3, y = (
            x1.to(device), x2.to(device), x3.to(device),
            y.to(device)
        )

        optimizer.zero_grad()

        outputs, pooled, y_pred = model([x1, x2, x3])

        loss = F.cross_entropy(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{(batch_idx+1)*len(x1)}/{len(train_loader.dataset)} "
                f"({100.*batch_idx/len(train_loader):.2f}%)]\tLoss: {loss.item():.6f}"
            )

    epoch_loss = running_loss / max(1, batches)
    print(f"===> Epoch {epoch} — Train Loss Promedio: {epoch_loss:.4f}")
    return epoch_loss


############################################
# VALIDATION FUNCTION — returns metrics + y_true/y_pred
############################################
def validation(model, device, test_loader):
    model.eval()
    total_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x1, x2, x3, y in test_loader:
            x1, x2, x3, y = (
                x1.to(device), x2.to(device), x3.to(device),
                y.to(device)
            )

            outputs, pooled, y_out = model([x1, x2, x3])

            loss = F.cross_entropy(y_out, y.squeeze())
            total_loss += loss.item()

            preds = torch.argmax(y_out, dim=1).cpu().numpy()
            labels = y.squeeze().cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

    avg_loss = total_loss / max(1, len(test_loader))

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"Validation → Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%, "
          f"Prec: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")

    # devolvemos también y_true/y_pred para la matriz de confusión
    return avg_loss, acc, prec, rec, f1, y_true, y_pred


############################################
# MAIN
############################################
def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#######---------------------> Usando dispositivo:", DEVICE)

    input_ids = []
    input_types = []
    input_masks = []
    label = []

    # ===== Cargar dataset =====
    dataPreprocess_bert("benign_urls.txt",  input_ids, input_types, input_masks, label, 0)
    dataPreprocess_bert("malware_urls.txt", input_ids, input_types, input_masks, label, 1)

    (
        input_ids_train, input_types_train, input_masks_train, y_train,
        input_ids_val,   input_types_val,   input_masks_val,   y_val
    ) = spiltDatast_bert(input_ids, input_types, input_masks, label)

    # ===== DataLoaders =====
    BATCH_SIZE = 64    # ajusta según VRAM

    train_data = TensorDataset(
        torch.tensor(input_ids_train),
        torch.tensor(input_types_train),
        torch.tensor(input_masks_train),
        torch.tensor(y_train)
    )
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)

    val_data = TensorDataset(
        torch.tensor(input_ids_val),
        torch.tensor(input_types_val),
        torch.tensor(input_masks_val),
        torch.tensor(y_val)
    )
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

    # ===== Modelo =====
    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    # ===== Entrenamiento =====
    NUM_EPOCHS = 5
    PATH = 'R_model/model.pth'
    best_acc = 0.0

    # para graficar
    history = []
    last_y_true = None
    last_y_pred = None

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
        val_loss, acc, prec, rec, f1, y_true_epoch, y_pred_epoch = validation(model, DEVICE, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        # guardamos las predicciones del ÚLTIMO epoch
        last_y_true = y_true_epoch
        last_y_pred = y_pred_epoch

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)
            print(f"==> Nuevo mejor modelo guardado. ACC = {best_acc:.4f}")

    print("\n==== ENTRENAMIENTO FINALIZADO ====\n")

    # ===== Guardar historial a CSV =====
    df = pd.DataFrame(history)
    df.to_csv("training_metrics.csv", index=False)
    print("Historial guardado en training_metrics.csv")

    # ===== Graficar curvas =====

    # --- Loss ---
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker='o')
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de Pérdida")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png")

    # --- Métricas ---
    plt.figure()
    plt.plot(df["epoch"], df["accuracy"], label="Accuracy", marker='o')
    plt.plot(df["epoch"], df["precision"], label="Precision", marker='o')
    plt.plot(df["epoch"], df["recall"], label="Recall", marker='o')
    plt.plot(df["epoch"], df["f1"], label="F1", marker='o')
    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.title("Curva de Métricas por Época")
    plt.grid(True)
    plt.legend()
    plt.savefig("metrics_curve.png")

    print("Gráficos generados: loss_curve.png y metrics_curve.png")

    # ===== Matriz de confusión (último epoch) =====
    if last_y_true is not None and last_y_pred is not None:
        cm = confusion_matrix(last_y_true, last_y_pred)
        print("Matriz de confusión (último epoch):\n", cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Real 0", "Real 1"],
        )
        plt.title("Matriz de Confusión (Último Epoch)")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("Matriz de confusión guardada como confusion_matrix.png")
    else:
        print("No se pudo generar la matriz de confusión (no hay datos).")


if __name__ == '__main__':
    main()
