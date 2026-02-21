import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve
)
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    model_save_path,
):
    """
    Loop principal de treinamento.

    Salva:
    - Loss
    - AUC
    - F1
    - Recall
    - Precision
    - Confusion matrix
    - ROC curve (FPR, TPR)

    Mantém compatibilidade com notebooks anteriores.
    """

    best_auc = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
        "val_recall": [],
        "val_precision": [],
        "val_confusion_matrix": [],
        "val_fpr": [],
        "val_tpr": []
    }

    for epoch in range(epochs):

        # ======================
        # TREINO
        # ======================
        model.train()
        running_train_loss = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Treino]",
            leave=False
        )

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_train_loss / len(train_loader)

        # ======================
        # VALIDAÇÃO
        # ======================
        model.eval()
        running_val_loss = 0

        all_labels = []
        all_probs = []

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Validação]",
            leave=False
        )

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                val_bar.set_postfix(loss=loss.item())

        val_loss = running_val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_probs)

        preds = [1 if p > 0.5 else 0 for p in all_probs]

        val_f1 = f1_score(all_labels, preds)
        val_recall = recall_score(all_labels, preds)
        val_precision = precision_score(all_labels, preds)
        val_conf = confusion_matrix(all_labels, preds)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)
        history["val_precision"].append(val_precision)
        history["val_confusion_matrix"].append(val_conf)
        history["val_fpr"].append(fpr)
        history["val_tpr"].append(tpr)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")
        print(f"Val F1:     {val_f1:.4f}")
        print(f"Val Recall: {val_recall:.4f}")
        print(f"Val Prec.:  {val_precision:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print("→ Novo melhor modelo salvo.")

    print("\nMelhor AUC obtida:", best_auc)

    return history, best_auc