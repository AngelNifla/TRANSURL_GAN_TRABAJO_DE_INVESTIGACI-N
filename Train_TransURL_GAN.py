# Train_TransURL_GAN.py

import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from gan_models import TransURLDiscriminator, AdversarialGenerator
from data_processing import (
    dataPreprocess_bert,
    spiltDatast_bert,
)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#######---------------------> Usando dispositivo (GAN-TransURL):", device)
    return device


def build_train_loader(batch_size):
    """
    Copiamos la lógica de main() de tu Train.py,
    pero solo necesitamos el loader de entrenamiento.
    """
    input_ids = []
    input_types = []
    input_masks = []
    label = []

    dataPreprocess_bert("benign_urls.txt", input_ids, input_types, input_masks, label, 0)
    dataPreprocess_bert("malware_urls.txt", input_ids, input_types, input_masks, label, 1)

    input_ids_train, input_types_train, input_masks_train, y_train, \
    input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast_bert(
        input_ids, input_types, input_masks, label
    )

    train_data = TensorDataset(
        torch.tensor(input_ids_train),
        torch.tensor(input_types_train),
        torch.tensor(input_masks_train),
        torch.tensor(y_train)
    )

    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_loader


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_training_curves(epochs,
                         loss_D_list,
                         loss_G_list,
                         acc_list,
                         prec_list,
                         rec_list,
                         f1_list,
                         adv_auc_list,
                         adv_acc_list,
                         feat_dist_list,
                         cm_last,
                         adv_logits_real_last,
                         adv_logits_fake_last,
                         features_real_last,
                         features_fake_last,
                         save_dir):
    graph_dir = os.path.join(save_dir, "result_graficos")
    ensure_dir(graph_dir)

    # 1) Curvas de pérdidas
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_D_list, label="Loss_D")
    plt.plot(epochs, loss_G_list, label="Loss_G")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Pérdidas del Discriminador y Generador")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "loss_D_G.png"))
    plt.close()

    # 2) Métricas de clasificación
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc_list, label="Accuracy")
    plt.plot(epochs, prec_list, label="Precision")
    plt.plot(epochs, rec_list, label="Recall")
    plt.plot(epochs, f1_list, label="F1-score")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title("Métricas de clasificación (benigno/malware)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "cls_metrics.png"))
    plt.close()

    # 3) Métricas adversariales
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, adv_auc_list, label="AUC real/fake")
    plt.plot(epochs, adv_acc_list, label="Accuracy real/fake")
    plt.plot(epochs, feat_dist_list, label="Distancia media features")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title("Métricas adversariales (Discriminador vs Generador)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "adv_metrics.png"))
    plt.close()

    # 4) Matriz de confusión (última época)
    if cm_last is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_last,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    xticklabels=['benign', 'malware'],
                    yticklabels=['benign', 'malware'])
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.title("Matriz de confusión (última época)")
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, "confusion_matrix_last_epoch.png"))
        plt.close()

    # 5) Histograma de logits real vs fake (última época)
    if adv_logits_real_last is not None and adv_logits_fake_last is not None:
        plt.figure(figsize=(8, 6))
        plt.hist(adv_logits_real_last, bins=30, alpha=0.6, label="Real")
        plt.hist(adv_logits_fake_last, bins=30, alpha=0.6, label="Fake")
        plt.xlabel("Logits real/fake")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de logits (real vs fake) - última época")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, "adv_logits_hist_last_epoch.png"))
        plt.close()

    # 6) PCA 2D de features reales vs fake
    if features_real_last is not None and features_fake_last is not None:
        X_real = np.array(features_real_last)
        X_fake = np.array(features_fake_last)

        max_points = 500
        if X_real.shape[0] > max_points:
            idx = np.random.choice(X_real.shape[0], max_points, replace=False)
            X_real = X_real[idx]
        if X_fake.shape[0] > max_points:
            idx = np.random.choice(X_fake.shape[0], max_points, replace=False)
            X_fake = X_fake[idx]

        X = np.vstack([X_real, X_fake])
        y = np.array([1] * X_real.shape[0] + [0] * X_fake.shape[0])  # 1=real, 0=fake

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        X_real_pca = X_pca[y == 1]
        X_fake_pca = X_pca[y == 0]

        plt.figure(figsize=(8, 6))
        plt.scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.6, label="Real", s=10)
        plt.scatter(X_fake_pca[:, 0], X_fake_pca[:, 1], alpha=0.6, label="Fake", s=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA de features reales vs fake (última época)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, "pca_features_last_epoch.png"))
        plt.close()


def train_gan(
    D,
    G,
    train_loader,
    device,
    num_epochs=3,
    lambda_cls=1.0,
    lambda_robust=1.0,
    lambda_adv=1.0,
    lr_D=2e-5,
    lr_G=1e-4,
    save_dir="R_model_gan"
):
    ensure_dir(save_dir)

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    optimizer_D = optim.Adam(D.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G)

    D.to(device)
    G.to(device)

    start_time = time.time()

    epochs_list = []
    loss_D_list = []
    loss_G_list = []
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    adv_auc_list = []
    adv_acc_list = []
    feat_dist_list = []

    cm_last = None
    adv_logits_real_last = None
    adv_logits_fake_last = None
    features_real_last = None
    features_fake_last = None

    for epoch in range(1, num_epochs + 1):
        D.train()
        G.train()

        running_loss_D = 0.0
        running_loss_G = 0.0

        all_y_true_cls = []
        all_y_pred_cls = []

        all_adv_logits = []
        all_adv_labels = []

        feat_dist_batches = []

        epoch_features_real = []
        epoch_features_fake = []

        for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device)

            batch_size = x1.size(0)

            y_cls = y.squeeze()
            if y_cls.dim() == 0:
                y_cls = y_cls.unsqueeze(0)

            # 1) Discriminador
            optimizer_D.zero_grad()

            cls_logits_real, adv_logits_real, pooled_real = D([x1, x2, x3])

            # Clasificación en ejemplos limpios
            loss_cls_real = ce_loss(cls_logits_real, y_cls)

            # Real/Fake en ejemplos reales
            real_targets = torch.ones_like(adv_logits_real, device=device)
            loss_adv_real = bce_loss(adv_logits_real, real_targets)

            # Generamos adversariales en espacio de features
            z = torch.randn(batch_size, G.latent_dim, device=device)
            h_adv, delta = G(pooled_real.detach(), y_cls, z)

            # Real/Fake en ejemplos adversariales
            adv_logits_fake = D.adv_head(h_adv)
            fake_targets = torch.zeros_like(adv_logits_fake, device=device)
            loss_adv_fake = bce_loss(adv_logits_fake, fake_targets)

            # NUEVO: clasificación también sobre h_adv (robustez)
            cls_logits_adv = D.cls_head(h_adv.detach())
            loss_cls_adv = ce_loss(cls_logits_adv, y_cls)

            # Loss total del Discriminador
            loss_D = (
                lambda_cls    * loss_cls_real +
                lambda_robust * loss_cls_adv +
                lambda_adv    * (loss_adv_real + loss_adv_fake)
            )
            loss_D.backward()
            optimizer_D.step()

            running_loss_D += loss_D.item()

            # Métricas clasificación (en ejemplos limpios)
            with torch.no_grad():
                preds_real = cls_logits_real.argmax(dim=-1)
                y_true_batch = y_cls.detach().cpu().numpy()
                y_pred_batch = preds_real.detach().cpu().numpy()
                all_y_true_cls.extend(y_true_batch.tolist())
                all_y_pred_cls.extend(y_pred_batch.tolist())

            # Métricas real/fake
            with torch.no_grad():
                adv_real_np = adv_logits_real.detach().cpu().numpy().reshape(-1)
                adv_fake_np = adv_logits_fake.detach().cpu().numpy().reshape(-1)
                all_adv_logits.extend(adv_real_np.tolist())
                all_adv_labels.extend([1] * len(adv_real_np))
                all_adv_logits.extend(adv_fake_np.tolist())
                all_adv_labels.extend([0] * len(adv_fake_np))

            # Distancia media features
            with torch.no_grad():
                min_len = min(pooled_real.size(0), h_adv.size(0))
                if min_len > 0:
                    dist = torch.norm(
                        pooled_real[:min_len] - h_adv[:min_len],
                        dim=1
                    ).mean().item()
                    feat_dist_batches.append(dist)

            # Guardar para PCA en última época
            if epoch == num_epochs:
                epoch_features_real.append(pooled_real.detach().cpu().numpy())
                epoch_features_fake.append(h_adv.detach().cpu().numpy())

            # 2) Generador
            optimizer_G.zero_grad()

            z2 = torch.randn(batch_size, G.latent_dim, device=device)
            h_adv_for_G, delta_for_G = G(pooled_real.detach(), y_cls, z2)

            adv_logits_fake_for_G = D.adv_head(h_adv_for_G)

            real_targets_for_G = torch.ones_like(adv_logits_fake_for_G, device=device)
            loss_G = lambda_adv * bce_loss(adv_logits_fake_for_G, real_targets_for_G)

            loss_G.backward()
            optimizer_G.step()

            running_loss_G += loss_G.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"[D] Loss_D: {loss_D.item():.4f}  "
                    f"[G] Loss_G: {loss_G.item():.4f}"
                )

        # Fin de época: métricas
        epoch_time = (time.time() - start_time) / 3600.0
        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)

        acc = accuracy_score(all_y_true_cls, all_y_pred_cls)
        prec = precision_score(all_y_true_cls, all_y_pred_cls)
        rec = recall_score(all_y_true_cls, all_y_pred_cls)
        f1 = f1_score(all_y_true_cls, all_y_pred_cls)
        cm = confusion_matrix(all_y_true_cls, all_y_pred_cls)

        adv_auc = None
        adv_acc = None
        try:
            adv_auc = roc_auc_score(all_adv_labels, all_adv_logits)
        except Exception as e:
            print(f"[Aviso] No se pudo calcular AUC real/fake en epoch {epoch}: {e}")
            adv_auc = float("nan")

        adv_preds_bin = (np.array(all_adv_logits) > 0.0).astype(int)
        adv_acc = accuracy_score(all_adv_labels, adv_preds_bin)

        if len(feat_dist_batches) > 0:
            feat_dist_epoch = float(np.mean(feat_dist_batches))
        else:
            feat_dist_epoch = float("nan")

        epochs_list.append(epoch)
        loss_D_list.append(avg_loss_D)
        loss_G_list.append(avg_loss_G)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        adv_auc_list.append(adv_auc)
        adv_acc_list.append(adv_acc)
        feat_dist_list.append(feat_dist_epoch)

        cm_last = cm
        adv_logits_real_last = np.array([v for v, lbl in zip(all_adv_logits, all_adv_labels) if lbl == 1])
        adv_logits_fake_last = np.array([v for v, lbl in zip(all_adv_logits, all_adv_labels) if lbl == 0])

        if epoch == num_epochs:
            if len(epoch_features_real) > 0:
                features_real_last = np.concatenate(epoch_features_real, axis=0)
            else:
                features_real_last = None
            if len(epoch_features_fake) > 0:
                features_fake_last = np.concatenate(epoch_features_fake, axis=0)
            else:
                features_fake_last = None

        print(f"\n==> Epoch {epoch}/{num_epochs} completada. Tiempo acum: {epoch_time:.2f} horas.")
        print(
            f"   [D] Loss_D: {avg_loss_D:.4f} | "
            f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f} | "
            f"AUC(real/fake): {adv_auc:.4f}  Acc(real/fake): {adv_acc:.4f} | "
            f"Dist_feat: {feat_dist_epoch:.4f}"
        )
        print(
            f"   [G] Loss_G: {avg_loss_G:.4f}"
        )

        torch.save(D.state_dict(), os.path.join(save_dir, f"D_epoch{epoch}.pt"))
        torch.save(G.state_dict(), os.path.join(save_dir, f"G_epoch{epoch}.pt"))

    # Gráficos
    plot_training_curves(
        epochs=epochs_list,
        loss_D_list=loss_D_list,
        loss_G_list=loss_G_list,
        acc_list=acc_list,
        prec_list=prec_list,
        rec_list=rec_list,
        f1_list=f1_list,
        adv_auc_list=adv_auc_list,
        adv_acc_list=adv_acc_list,
        feat_dist_list=feat_dist_list,
        cm_last=cm_last,
        adv_logits_real_last=adv_logits_real_last,
        adv_logits_fake_last=adv_logits_fake_last,
        features_real_last=features_real_last,
        features_fake_last=features_fake_last,
        save_dir=save_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--save_dir", type=str, default="R_model_gan")
    parser.add_argument("--pretrained_transurl", type=str, default="R_model/model.pth",
                        help="Ruta al .pth de TransURL preentrenado (baseline)")
    args = parser.parse_args()

    device = get_device()

    train_loader = build_train_loader(args.batch_size)

    D = TransURLDiscriminator(
        hidden_dim=args.hidden_dim,
        pretrained_weights_path=args.pretrained_transurl
    )

    G = AdversarialGenerator(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=2,
        class_emb_dim=16,
        epsilon=1.0
    )

    train_gan(
        D=D,
        G=G,
        train_loader=train_loader,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
