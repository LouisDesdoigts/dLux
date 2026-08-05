# Sparse

## Inheritance

```mermaid
classDiagram
    class dLux_layers_sparse_Interfere["Interfere"]
    class dLux_layers_sparse_SparseOptic["SparseOptic"]
    class dLux_layers_sparse_SparseDynamicOptic["SparseDynamicOptic"]
    class dLux_layers_dynamic_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_optical_Optic["Optic"]
    class dLux_layers_optical_OpticalLayer["OpticalLayer"]
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_sparse_Interfere
    click dLux_layers_sparse_Interfere href "#dLux.layers.sparse.Interfere" "Methods: apply()"
    dLux_layers_optical_Optic <|-- dLux_layers_sparse_SparseOptic
    click dLux_layers_sparse_SparseOptic href "#dLux.layers.sparse.SparseOptic" "Attributes: transmission, opd, phase, normalise, centers · Properties: n_apertures · Methods: phasor(), localise()"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_sparse_SparseDynamicOptic
    dLux_layers_sparse_SparseOptic <|-- dLux_layers_sparse_SparseDynamicOptic
    click dLux_layers_sparse_SparseDynamicOptic href "#dLux.layers.sparse.SparseDynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise, centers"
```

???+ info "Interfere"
    ::: dLux.layers.sparse.Interfere

???+ info "SparseOptic"
    ::: dLux.layers.sparse.SparseOptic

???+ info "SparseDynamicOptic"
    ::: dLux.layers.sparse.SparseDynamicOptic
