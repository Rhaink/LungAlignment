#!/usr/bin/env python3
"""
Crear imagen de referencia clara con números de landmarks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis" / "figures" / "debug"

# Cargar datos
data = np.load(PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz", allow_pickle=True)
all_landmarks = data['all_landmarks']
all_image_names = data['all_image_names']
all_categories = data['all_categories']

# Buscar una imagen Normal clara
idx_normal = np.where(all_categories == 'Normal')[0][0]
img_name = all_image_names[idx_normal]
img_path = PROJECT_ROOT / "data" / "dataset" / "Normal" / f"{img_name}.png"

print(f"Cargando imagen: {img_path}")
img = Image.open(img_path).convert('L')
img_224 = img.resize((224, 224))
img_arr = np.array(img_224)

# Landmarks de esta imagen
landmarks = all_landmarks[idx_normal]

# Crear figura grande y clara
fig, ax = plt.subplots(figsize=(16, 16))

ax.imshow(img_arr, cmap='gray')

# Plotear cada landmark con número grande y claro
for i, (x, y) in enumerate(landmarks):
    # Círculo de fondo
    circle = plt.Circle((x, y), 8, color='yellow', zorder=4)
    ax.add_patch(circle)

    # Borde del círculo
    circle_border = plt.Circle((x, y), 8, color='black', fill=False, linewidth=2, zorder=5)
    ax.add_patch(circle_border)

    # Número
    ax.text(x, y, str(i+1), fontsize=11, fontweight='bold',
            ha='center', va='center', color='black', zorder=6)

ax.set_xlim(0, 224)
ax.set_ylim(224, 0)
ax.set_title(f'IMAGEN DE REFERENCIA: {img_name}\nIndicar conexiones de landmarks', fontsize=16, fontweight='bold')
ax.axis('off')

# Agregar leyenda a la derecha
legend_text = """LANDMARKS:
1: Superior (mediastino)
2: Inferior (vertebra)
3: Apex Izquierdo
4: Apex Derecho
5: Hilio Izquierdo
6: Hilio Derecho
7: Base Izquierda
8: Base Derecha
9: Centro Superior
10: Centro Medio
11: Centro Inferior
12: Borde Sup Izquierdo
13: Borde Sup Derecho
14: Costofrenico Izquierdo
15: Costofrenico Derecho

EJE CENTRAL:
1 → 9 → 10 → 11 → 2

PULMON IZQUIERDO:
¿12 → 3 → 5 → 7 → 14?

PULMON DERECHO:
¿13 → 4 → 6 → 8 → 15?
"""

ax.text(1.02, 0.95, legend_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='black'))

plt.tight_layout()
output_path = OUTPUT_DIR / "REFERENCIA_landmarks_numerados.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Guardado: {output_path}")
plt.close()

print("\nPor favor revisa esta imagen y confirma o corrige las conexiones.")
