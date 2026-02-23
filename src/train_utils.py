import torch
# Biblioteca principal para deep learning (PyTorch)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve
)
# Métricas utilizadas para avaliar desempenho do modelo na validação

from tqdm import tqdm
# Biblioteca para exibir barra de progresso durante treino/validação


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
    # Armazena a melhor AUC já obtida (usada para salvar o melhor modelo)

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
    # Dicionário que guarda todo o histórico de métricas ao longo das épocas

    for epoch in range(epochs):
        # Loop principal de épocas

        # ======================
        # TREINO
        # ======================
        model.train()
        # Coloca o modelo em modo treino (ativa dropout, batchnorm etc.)

        running_train_loss = 0
        # Acumulador da loss de treino

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Treino]",
            leave=False
        )
        # Barra de progresso para o loader de treino

        for images, labels in train_bar:
            # Percorre os batches do treino

            images = images.to(device)
            labels = labels.to(device)
            # Envia dados para CPU ou GPU

            optimizer.zero_grad()
            # Zera os gradientes acumulados

            outputs = model(images)
            # Forward pass (predição)

            loss = criterion(outputs, labels)
            # Calcula a loss comparando predição com rótulo real

            loss.backward()
            # Backpropagation (calcula gradientes)

            optimizer.step()
            # Atualiza os pesos do modelo

            running_train_loss += loss.item()
            # Soma a loss do batch atual

            train_bar.set_postfix(loss=loss.item())
            # Atualiza a barra mostrando a loss do batch

        train_loss = running_train_loss / len(train_loader)
        # Calcula a média da loss de treino na época

        # ======================
        # VALIDAÇÃO
        # ======================
        model.eval()
        # Coloca o modelo em modo avaliação (desativa dropout etc.)

        running_val_loss = 0
        # Acumulador da loss de validação

        all_labels = []
        all_probs = []
        # Listas para armazenar todos os rótulos e probabilidades

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Validação]",
            leave=False
        )
        # Barra de progresso para validação

        with torch.no_grad():
            # Desativa cálculo de gradiente (economiza memória)

            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # Forward pass

                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                # Soma a loss de validação

                probs = torch.softmax(outputs, dim=1)[:, 1]
                # Converte logits em probabilidades
                # Pega probabilidade da classe positiva (Pneumonia)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                # Guarda rótulos e probabilidades para cálculo das métricas

                val_bar.set_postfix(loss=loss.item())
                # Atualiza barra com loss do batch

        val_loss = running_val_loss / len(val_loader)
        # Média da loss de validação

        val_auc = roc_auc_score(all_labels, all_probs)
        # Calcula AUC usando probabilidades contínuas

        preds = [1 if p > 0.5 else 0 for p in all_probs]
        # Converte probabilidades em classes usando threshold fixo 0.5

        val_f1 = f1_score(all_labels, preds)
        # Calcula F1-score

        val_recall = recall_score(all_labels, preds)
        # Calcula Recall (Sensitivity)

        val_precision = precision_score(all_labels, preds)
        # Calcula Precision

        val_conf = confusion_matrix(all_labels, preds)
        # Matriz de confusão (TP, TN, FP, FN)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        # Calcula pontos da curva ROC

        # Armazena tudo no histórico
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)
        history["val_precision"].append(val_precision)
        history["val_confusion_matrix"].append(val_conf)
        history["val_fpr"].append(fpr)
        history["val_tpr"].append(tpr)

        # Impressão organizada da época
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")
        print(f"Val F1:     {val_f1:.4f}")
        print(f"Val Recall: {val_recall:.4f}")
        print(f"Val Prec.:  {val_precision:.4f}")

        # Salva o modelo se a AUC for a melhor até agora
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print("→ Novo melhor modelo salvo.")

    print("\nMelhor AUC obtida:", best_auc)
    # Mostra a melhor AUC final

    return history, best_auc
    # Retorna histórico completo e melhor AUC