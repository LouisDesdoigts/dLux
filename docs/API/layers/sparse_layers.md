# Sparse Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_sparse_layers_Interfere["Interfere"]
    class dLux_layers_sparse_layers_SparseOptic["SparseOptic"]
    class dLux_layers_sparse_layers_SparseDynamicOptic["SparseDynamicOptic"]
    class dLux_layers_dynamic_layers_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_optical_layers_Optic["Optic"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_sparse_layers_Interfere
    click dLux_layers_sparse_layers_Interfere href "#dLux.layers.sparse_layers.Interfere" "No direct public attributes or methods"
    dLux_layers_optical_layers_Optic <|-- dLux_layers_sparse_layers_SparseOptic
    click dLux_layers_sparse_layers_SparseOptic href "#dLux.layers.sparse_layers.SparseOptic" "Attributes: transmission, opd, phase, normalise, centers · Properties: n_apertures · Methods: phasor(), localise()"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_sparse_layers_SparseDynamicOptic
    dLux_layers_sparse_layers_SparseOptic <|-- dLux_layers_sparse_layers_SparseDynamicOptic
    click dLux_layers_sparse_layers_SparseDynamicOptic href "#dLux.layers.sparse_layers.SparseDynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise, centers"
```

???+ info "Interfere"
    ::: dLux.layers.sparse_layers.Interfere

???+ info "SparseOptic"
    ::: dLux.layers.sparse_layers.SparseOptic

???+ info "SparseDynamicOptic"
    ::: dLux.layers.sparse_layers.SparseDynamicOptic
