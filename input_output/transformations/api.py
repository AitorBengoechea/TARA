"""Optional preprocessing of native engine input files.

Nothing in the current TARA execution pipeline calls this function.
"""
from pathlib import Path
from input_output.transformations.models import InputTransformationError
from input_output.transformations.registry import adapter_for


def transform_inputs(*, source, template, parameters, output):
    """Transfer selected physical parameters into a copy of a native template.

    The source and template may belong to the same or different engines.
    Each adapter converts native content to/from an engine-independent
    parameter. Existing LEAPR liquid diffusion, oscillator and scattering
    cards, as well as other NCMAT DYNINFO blocks, are left unchanged.

    The target must be a distinct path. On any error no output is written.
    Returns pathlib.Path for the created native input file. No report created.
    """
    src, target, dest = map(Path, (source, template, output))
    if src.resolve() == dest.resolve() or target.resolve() == dest.resolve():
        raise InputTransformationError('Output must differ from both source and template.')
    if isinstance(parameters, str):
        raise InputTransformationError('parameters must be a sequence, for example ["H_pdos"].')
    names = list(parameters)
    if not names or len(set(names)) != len(names):
        raise InputTransformationError('At least one distinct parameter is required.')
    source_adapter = adapter_for(src)
    target_adapter = adapter_for(target)
    if adapter_for(dest).engine != target_adapter.engine:
        raise InputTransformationError('Output format must match the template format.')
    source_text = src.read_text(encoding='utf-8')
    output_text = target.read_text(encoding='utf-8')
    for name in names:
        physical_value = source_adapter.extract(source_text, name)
        output_text = target_adapter.replace(output_text, name, physical_value)
    # Validate the final native format using its adapter before publishing.
    for name in names:
        target_adapter.extract(output_text, name)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing generated input: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output_text, encoding='utf-8')
    return dest


def available_parameters(input_path):
    """List named transferable parameters actually present in a native input.

    For NCMAT this is determined from @DYNINFO element/type fields, rather
    than from hardcoded species names. LEAPR exposes its principal species
    only when the native AWR unambiguously identifies the element.
    """
    path = Path(input_path)
    adapter = adapter_for(path)
    if not hasattr(adapter, 'available_parameters'):
        raise InputTransformationError(
            f'Adapter {adapter.engine!r} does not support parameter discovery.')
    return adapter.available_parameters(path.read_text(encoding='utf-8'))
