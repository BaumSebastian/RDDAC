# Preprocessing

The supported interface is the CLI, [`rddac preprocess`](../cli.md#rddac-preprocess); what each stage does is documented in the [Preprocessing section](../preprocessing/index.md). The Python functions below live in the private `rddac._preprocess` package and are listed for orientation (notebooks, custom variants), not as a stable API.

!!! warning "Internal API, may change without notice"
    Anyone replacing a stage should follow [Custom processing](../preprocessing/custom.md): read raw data via the public API and match the processed schema, instead of importing from here.

## Stages

::: rddac._preprocess.force.process
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.sheet.process
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.oil.process
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.oil.hampel_filter
    options:
      show_signature: true
      show_signature_annotations: true

The `pointcloud` stage owns its whole file section (`rddac._preprocess.pointcloud.stage.process_experiment`) and needs the `[preprocessing]` extra plus the DDACS simulations; see [Point Clouds](../preprocessing/pointcloud.md).

## Orchestration and configuration

::: rddac._preprocess.runner.run
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.config.load
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.config.dump
    options:
      show_signature: true

## Processing figures

The figures on the preprocessing pages are drawn from raw arrays by these functions; `python -m rddac._preprocess.visualize --help` renders them to files.

::: rddac._preprocess.visualize.plot_oil_processing
    options:
      show_signature: true

::: rddac._preprocess.visualize.plot_force_processing
    options:
      show_signature: true

::: rddac._preprocess.visualize.plot_sheet_processing
    options:
      show_signature: true

::: rddac._preprocess.visualize.plot_luminescence_processing
    options:
      show_signature: true

::: rddac._preprocess.visualize.plot_pointcloud_processing
    options:
      show_signature: true
