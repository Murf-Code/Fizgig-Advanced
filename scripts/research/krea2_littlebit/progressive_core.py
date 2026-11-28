"""Fixed H10 phase schedule; caller supplies the existing H5 optimizer."""

ARMS = ('joint', 'joint-reset', 'progressive')


def run_schedule(name, make_primary, make_residual, optimize, record):
    """100 updates in every arm; staged arms reset Adam and row RNG at 40.

    Callbacks are phase-local; translate them to global steps here. Only the
    progressive arm delays construction of the second path until update 40.
    The caller retains original teacher and calibration energy normalization.
    """
    if name not in ARMS:
        raise ValueError('unknown H10 arm')
    paths = [make_primary()]
    if name != 'progressive':
        paths.append(make_residual(paths[0]))
    record(paths, {'step': 0, 'phase': 'initial'})

    def phase(label, offset, steps, warmup):
        def callback(ps, stats):
            record(ps, {**stats, 'step': offset + stats['step'], 'phase': label})
        optimize(paths, steps, warmup, callback)

    if name == 'joint':
        phase('joint', 0, 100, 20)
    else:
        phase('primary' if name == 'progressive' else 'joint_before_reset', 0, 40, 20)
        if name == 'progressive':
            paths.append(make_residual(paths[0]))
            record(paths, {'step': 40, 'phase': 'residual_initialized'})
        phase('joint_after_reset', 40, 60, 0)
    return paths
