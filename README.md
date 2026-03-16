NuLite con postprocessing di CellVit++ tramite GPU.

Ho usato come base NuLite (https://github.com/CosmoIknosLab/NuLite), e ho modificato alcune parti di codice in base al post-pricessing CPU di CellViT++ (https://github.com/TIO-IKIM/CellViT). Le modifiche apportate sono le seguenti:

1. Copiato il file "cellvit/models/cell_segmentation/postprocessing.py" in NuLite e cambiato l'import in nulite.py:
```python
# PRIMA
from nuclei_detection.utils.post_proc_nulite import DetectionCellPostProcessor

# DOPO
from nuclei_detection.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
```

2. Modificata in nulite.py la funzione calculate_instance_map.py:
```python
# PRIMA
cell_post_processor = DetectionCellPostProcessor(
    nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
)
...
instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
instance_preds.append(instance_pred[0])
type_preds.append(instance_pred[1])

# DOPO
cell_post_processor = DetectionCellPostProcessorCupy(
    wsi=wsi, nr_types=self.num_nuclei_classes,
    resolution=0.25 if magnification == 40 else 0.5, gt=False
)
...
pred_inst, pred_type = cell_post_processor.post_process_single_image(
    cp.asarray(pred_map)
)
cells = cell_post_processor._create_cell_dict(pred_inst, pred_type)
instance_preds.append(pred_inst)
type_preds.append(cells)
```

3. Aggiunta di wsi=None
```python
def calculate_instance_map(self, predictions: OrderedDict, magnification: Literal[20, 40] = 20, wsi=None)
```

4. Per evitare problemi con cupy, bisogna eseguire "export LD_LIBRARY_PATH=/opt/conda/envs/NuCelGPUProva/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

5. Modificato in nulite.py il loop come segue:
```python
for i in range(predictions_["nuclei_binary_map"].shape[0]):
    pred_map = np.concatenate(
        [
            torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i].detach().cpu()[..., None],
            torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i].detach().cpu()[..., None],
            predictions_["hv_map"][i].detach().cpu(),
        ],
        axis=-1,
    )
    pred_inst, cells = cell_post_processor.post_process_single_image(
        cp.asarray(pred_map)
    )
    instance_preds.append(pred_inst)
    type_preds.append(cells)
```

6. Aggiunta la funzione remove_small_objects_cp in nuclei_detection/utils/tools.py

7. Cambiati gli import in postprocessing_cupy.py
```python
from nuclei_detection.datamodel.wsi_datamodel import WSI
from nuclei_detection.utils.tools import get_bounding_box, remove_small_objects_cp
from nuclei_detection.utils.metrics import remap_label
```

8. Sostituito in postprocessing_cupy.py 
```python
# PRIMA
wsi: Union[WSI,WSIMetadata]

# DOPO
wsi: WSI
```


Nota a margine. Per verificare che effettivamente sia usato il post processing GPU in postprocess_cupy.py è stata aggiunta la seguente riga:
```python
print(f"[GPU POST-PROC] input type: {type(pred_inst)}, is cupy: {isinstance(pred_inst, cp.ndarray)}")
```
