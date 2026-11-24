"""Optional preprocessing of native engine input files.

Nothing in the current TARA execution pipeline calls this function.
"""
from pathlib import Path
from input_output.transformations.models import InputTransformationError
from input_output.transformations.registry import adapter_for


def transform_inputs(*, source, template, parameters, output):
    """Transfer selected physical parameters into a copy of a native template.

    Parameters
    ----------
    source, template : str or pathlib.Path
        Paths to the parameter source and target template.
    parameters : sequence of str
        Distinct parameter names exported by the source adapter.
    output : str or pathlib.Path
        New output path using the template format.

    Returns
    -------
    pathlib.Path
        Path of the generated native input file.

    Raises
    ------
    InputTransformationError
        If the selected parameters or output format are incompatible.
    FileExistsError
        If the output path already exists.
    """
    src, target, dest = map(Path, (source, template, output))

    # Resolve aliases and symlinks to protect both original input files.
    if src.resolve() == dest.resolve() or target.resolve() == dest.resolve():
        raise InputTransformationError('Output must differ from both source and template.')

    # Reject a bare string before it can be split into individual characters.
    if isinstance(parameters, str):
        raise InputTransformationError('parameters must be a sequence, for example ["H_pdos"].')
    names = list(parameters)
    if not names or len(set(names)) != len(names):
        raise InputTransformationError('At least one distinct parameter is required.')

    # Read with each input's adapter and keep the generated file in the template's format.
    source_adapter = adapter_for(src)
    target_adapter = adapter_for(target)
    if adapter_for(dest).engine != target_adapter.engine:
        raise InputTransformationError('Output format must match the template format.')

    # Apply and validate all replacements in memory before writing the new file.
    source_text = src.read_text(encoding='utf-8')
    output_text = target.read_text(encoding='utf-8')
    output_text = _transfer_parameters(source_adapter, target_adapter, source_text, output_text, names)

    # Preserve any previously generated file at the destination.
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing generated input: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output_text, encoding='utf-8')
    return dest


def _transfer_parameters(source_adapter, target_adapter, source_text, target_text, names):
    """Replace the named parameters and validate the final native content.

    Parameters
    ----------
    source_adapter, target_adapter : LeaprAdapter or NCrystalAdapter
        Adapters for the source and target input formats.
    source_text, target_text : str
        Native input content for the source and target.
    names : sequence of str
        Parameter names in the order in which to transfer them.

    Returns
    -------
    str
        Native target content with the selected parameters replaced.
    """
    for name in names:
        physical_value = source_adapter.extract(source_text, name)
        target_text = target_adapter.replace(target_text, name, physical_value)

    # Read all replaced values back after the final edit to catch interactions.
    for name in names:
        target_adapter.extract(target_text, name)
    return target_text


def available_parameters(input_path):
    """List named transferable parameters present in a native input.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Source NCMAT or LEAPR path.

    Returns
    -------
    list of str
        Parameters discoverable by the selected format adapter.

    Raises
    ------
    InputTransformationError
        If the format or parameter discovery is unsupported.
    """
    path = Path(input_path)
    adapter = adapter_for(path)
    if not hasattr(adapter, 'available_parameters'):
        raise InputTransformationError(
            f'Adapter {adapter.engine!r} does not support parameter discovery.')
    return adapter.available_parameters(path.read_text(encoding='utf-8'))
