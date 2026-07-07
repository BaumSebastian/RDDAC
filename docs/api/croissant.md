# Croissant

`rddac.load` and `rddac.add_view` are the two entry points to the Croissant manifest. Both are re-exported at the top level (`rddac.load`, `rddac.add_view`); the implementation lives in `rddac.croissant`.

## `rddac.load`

::: rddac.croissant.load
    options:
      show_signature: true
      show_signature_annotations: true

## `rddac.add_view`

::: rddac.croissant.add_view
    options:
      show_signature: true
      show_signature_annotations: true

## Module reference

::: rddac.croissant
    options:
      members:
        - metadata_url
        - resolve_source
        - field_map
        - process_parameters_descriptions
        - dataset_name
      show_root_heading: false
      heading_level: 3

`rddac.croissant.METADATA_URL` mirrors the DDACS constant of the same name, but is a lazy module attribute: DaRUS assigns no per-file persistent ids, so the numeric file id behind the download URL is resolved through the DaRUS API on first access (see `metadata_url` above) and cached for the process lifetime.
