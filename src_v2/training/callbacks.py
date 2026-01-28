"""
Callbacks para entrenamiento
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np


logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping basado en metrica de validacion.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Epocas a esperar sin mejora
            min_delta: Cambio minimo para considerar mejora
            mode: 'min' o 'max'
            verbose: Imprimir mensajes
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Actualiza estado y retorna si debe parar.

        Args:
            score: Metrica actual
            epoch: Epoca actual

        Returns:
            True si debe parar
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: %d/%d (best: %.4f @ epoch %d)",
                    self.counter, self.patience, self.best_score, self.best_epoch
                )

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered!")

        return self.should_stop

    def reset(self):
        """Reinicia el estado."""
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False


class ModelCheckpoint:
    """
    Guarda checkpoints del modelo.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: Directorio para guardar
            monitor: Metrica a monitorear
            mode: 'min' o 'max'
            save_best_only: Solo guardar si mejora
            verbose: Imprimir mensajes
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score = None
        self.best_path = None

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Optional[str]:
        """
        Guarda checkpoint si corresponde.

        Args:
            model: Modelo a guardar
            optimizer: Optimizador
            epoch: Epoca actual
            metrics: Dict con metricas

        Returns:
            Path del archivo guardado o None
        """
        score = metrics.get(self.monitor, None)
        if score is None:
            return None

        should_save = False
        if self.best_score is None:
            should_save = True
        elif self.mode == 'min' and score < self.best_score:
            should_save = True
        elif self.mode == 'max' and score > self.best_score:
            should_save = True

        if not should_save and self.save_best_only:
            return None

        # Actualizar mejor score
        if should_save:
            self.best_score = score

        # Guardar checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:03d}_{self.monitor}{score:.4f}_{timestamp}.pt"
        filepath = self.save_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }

        torch.save(checkpoint, filepath)

        if self.verbose and should_save:
            logger.info("Checkpoint saved: %s (%s=%.4f)", filepath.name, self.monitor, score)

        # Guardar referencia al mejor
        if should_save:
            self.best_path = filepath

        return str(filepath)

    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Carga el mejor checkpoint.

        Args:
            model: Modelo donde cargar
            optimizer: Optimizador opcional
        """
        if self.best_path is None or not self.best_path.exists():
            logger.warning("No checkpoint found to load")
            return

        checkpoint = torch.load(self.best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info("Loaded best checkpoint from epoch %d", checkpoint['epoch'])


class LRSchedulerCallback:
    """
    Wrapper para learning rate schedulers.
    """

    def __init__(self, scheduler, step_on: str = 'epoch'):
        """
        Args:
            scheduler: torch.optim.lr_scheduler
            step_on: 'epoch' o 'batch'
        """
        self.scheduler = scheduler
        self.step_on = step_on

    def step_epoch(self, metrics: Optional[Dict] = None):
        """Llamar al final de cada epoca."""
        if self.step_on == 'epoch':
            if hasattr(self.scheduler, 'step'):
                # Para ReduceLROnPlateau necesita metrica
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if metrics and 'val_loss' in metrics:
                        self.scheduler.step(metrics['val_loss'])
                else:
                    self.scheduler.step()

    def step_batch(self):
        """Llamar al final de cada batch."""
        if self.step_on == 'batch':
            self.scheduler.step()

    def get_last_lr(self):
        """Retorna ultimo learning rate."""
        return self.scheduler.get_last_lr()
