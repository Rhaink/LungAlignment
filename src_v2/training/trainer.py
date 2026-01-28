"""
Trainer unificado para landmark prediction
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src_v2.constants import DEFAULT_IMAGE_SIZE
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


logger = logging.getLogger(__name__)


class LandmarkTrainer:
    """
    Trainer para entrenamiento en dos fases:
    - Phase 1: Backbone congelado, entrenar solo cabeza
    - Phase 2: Fine-tuning completo con LR diferenciado
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str = './checkpoints',
        image_size: int = DEFAULT_IMAGE_SIZE
    ):
        """
        Args:
            model: Modelo ResNet18Landmarks
            device: Dispositivo (cuda/cpu)
            save_dir: Directorio para checkpoints
            image_size: Tamano de imagen para metricas en pixeles
        """
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_error_px': [],
            'val_error_px': [],
            'lr': []
        }

    def compute_pixel_error(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula error euclidiano promedio en pixeles.

        Args:
            pred: Predicciones (B, 30) en [0, 1]
            target: Ground truth (B, 30) en [0, 1]

        Returns:
            Error promedio en pixeles
        """
        B = pred.shape[0]
        pred = pred.view(B, 15, 2) * self.image_size
        target = target.view(B, 15, 2) * self.image_size

        # Error euclidiano por landmark
        errors = torch.norm(pred - target, dim=-1)  # (B, 15)
        return errors.mean()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Callable,
        scheduler_callback: Optional[LRSchedulerCallback] = None
    ) -> Dict[str, float]:
        """
        Entrena una epoca.

        Returns:
            Dict con loss y error_px promedio
        """
        self.model.train()
        total_loss = 0.0
        total_error = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, landmarks, _ in pbar:
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            optimizer.zero_grad()

            # Forward
            outputs = self.model(images)

            # Loss
            if hasattr(criterion, 'forward') and callable(criterion):
                loss_dict = criterion(outputs, landmarks)
                if isinstance(loss_dict, dict):
                    loss = loss_dict['total']
                else:
                    loss = loss_dict
            else:
                loss = criterion(outputs, landmarks)

            # Backward
            loss.backward()
            optimizer.step()

            # Metricas
            with torch.no_grad():
                error_px = self.compute_pixel_error(outputs, landmarks)

            total_loss += loss.item()
            total_error += error_px.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'error': f'{error_px.item():.2f}px'
            })

            if scheduler_callback:
                scheduler_callback.step_batch()

        return {
            'loss': total_loss / num_batches,
            'error_px': total_error / num_batches
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Valida el modelo.

        Returns:
            Dict con val_loss y val_error_px
        """
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        num_batches = 0

        for images, landmarks, _ in val_loader:
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            outputs = self.model(images)

            # Loss
            if hasattr(criterion, 'forward') and callable(criterion):
                loss_dict = criterion(outputs, landmarks)
                if isinstance(loss_dict, dict):
                    loss = loss_dict['total']
                else:
                    loss = loss_dict
            else:
                loss = criterion(outputs, landmarks)

            error_px = self.compute_pixel_error(outputs, landmarks)

            total_loss += loss.item()
            total_error += error_px.item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_error_px': total_error / num_batches
        }

    def train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable,
        epochs: int = 15,
        lr: float = 1e-3,
        patience: int = 5
    ) -> Dict[str, List]:
        """
        Phase 1: Entrenar solo cabeza con backbone congelado.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validacion
            criterion: Funcion de perdida
            epochs: Numero de epocas
            lr: Learning rate para cabeza
            patience: Paciencia para early stopping

        Returns:
            Historial de entrenamiento
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Training head only (backbone + coord_attention frozen)")
        logger.info("=" * 60)

        # Asegurar backbone y coord_attention congelados
        if hasattr(self.model, 'freeze_all_except_head'):
            self.model.freeze_all_except_head()
        else:
            self.model.freeze_backbone()

        # Optimizador solo para cabeza
        optimizer = torch.optim.Adam(
            self.model.head.parameters(),
            lr=lr
        )

        # Callbacks
        early_stopping = EarlyStopping(patience=patience, mode='min')
        checkpoint = ModelCheckpoint(
            save_dir=self.save_dir / 'phase1',
            monitor='val_error_px',
            mode='min'
        )

        history = {'train_loss': [], 'val_loss': [], 'train_error_px': [], 'val_error_px': []}

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            elapsed = time.time() - start_time

            # Log
            logger.info(
                "Epoch %d/%d (%.1fs) - Train: loss=%.4f, error=%.2fpx - Val: loss=%.4f, error=%.2fpx",
                epoch + 1, epochs, elapsed,
                train_metrics['loss'], train_metrics['error_px'],
                val_metrics['val_loss'], val_metrics['val_error_px']
            )

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['train_error_px'].append(train_metrics['error_px'])
            history['val_error_px'].append(val_metrics['val_error_px'])

            # Callbacks
            all_metrics = {**train_metrics, **val_metrics}
            checkpoint(self.model, optimizer, epoch, all_metrics)

            if early_stopping(val_metrics['val_error_px'], epoch):
                break

        # Cargar mejor modelo
        checkpoint.load_best(self.model)
        logger.info("Phase 1 complete. Best val error: %.2fpx", early_stopping.best_score)

        return history

    def train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable,
        epochs: int = 100,
        backbone_lr: float = 2e-5,
        head_lr: float = 2e-4,
        patience: int = 10
    ) -> Dict[str, List]:
        """
        Phase 2: Fine-tuning completo con LR diferenciado.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validacion
            criterion: Funcion de perdida
            epochs: Numero de epocas
            backbone_lr: Learning rate para backbone
            head_lr: Learning rate para cabeza
            patience: Paciencia para early stopping

        Returns:
            Historial de entrenamiento
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-tuning (backbone + coord_attention unfrozen)")
        logger.info("=" * 60)

        # Descongelar todo (backbone + coord_attention)
        if hasattr(self.model, 'unfreeze_all'):
            self.model.unfreeze_all()
        else:
            self.model.unfreeze_backbone()

        # Optimizador con LR diferenciado
        param_groups = self.model.get_trainable_params()
        param_groups[0]['lr'] = backbone_lr  # features (backbone + coord_attention)
        param_groups[1]['lr'] = head_lr      # head

        optimizer = torch.optim.Adam(param_groups)

        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scheduler_callback = LRSchedulerCallback(scheduler, step_on='epoch')

        # Callbacks
        early_stopping = EarlyStopping(patience=patience, mode='min')
        checkpoint = ModelCheckpoint(
            save_dir=self.save_dir / 'phase2',
            monitor='val_error_px',
            mode='min'
        )

        history = {'train_loss': [], 'val_loss': [], 'train_error_px': [], 'val_error_px': [], 'lr': []}

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            # Log
            logger.info(
                "Epoch %d/%d (%.1fs) [lr=%.2e] - Train: loss=%.4f, error=%.2fpx - Val: loss=%.4f, error=%.2fpx",
                epoch + 1, epochs, elapsed, current_lr,
                train_metrics['loss'], train_metrics['error_px'],
                val_metrics['val_loss'], val_metrics['val_error_px']
            )

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['train_error_px'].append(train_metrics['error_px'])
            history['val_error_px'].append(val_metrics['val_error_px'])
            history['lr'].append(current_lr)

            # Callbacks
            all_metrics = {**train_metrics, **val_metrics}
            checkpoint(self.model, optimizer, epoch, all_metrics)
            scheduler_callback.step_epoch(all_metrics)

            if early_stopping(val_metrics['val_error_px'], epoch):
                break

        # Cargar mejor modelo
        checkpoint.load_best(self.model)
        logger.info("Phase 2 complete. Best val error: %.2fpx", early_stopping.best_score)

        return history

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable,
        phase1_epochs: int = 15,
        phase2_epochs: int = 50,
        phase1_lr: float = 1e-3,
        phase2_backbone_lr: float = 2e-5,
        phase2_head_lr: float = 2e-4,
        phase1_patience: int = 5,
        phase2_patience: int = 10
    ) -> Dict[str, List]:
        """
        Entrenamiento completo en dos fases.

        Returns:
            Historial combinado
        """
        # Phase 1
        history1 = self.train_phase1(
            train_loader, val_loader, criterion,
            epochs=phase1_epochs,
            lr=phase1_lr,
            patience=phase1_patience
        )

        # Phase 2 (optional)
        if phase2_epochs > 0:
            history2 = self.train_phase2(
                train_loader, val_loader, criterion,
                epochs=phase2_epochs,
                backbone_lr=phase2_backbone_lr,
                head_lr=phase2_head_lr,
                patience=phase2_patience
            )
        else:
            history2 = {'train_loss': [], 'val_loss': [], 'train_error_px': [], 'val_error_px': [], 'lr': []}

        # Combinar historiales
        combined = {
            'phase1': history1,
            'phase2': history2
        }

        return combined

    def save_model(self, path: str, include_optimizer: bool = False):
        """Guarda modelo final."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str):
        """Carga modelo guardado."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        logger.info("Model loaded from %s", path)
