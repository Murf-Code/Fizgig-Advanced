"""Fixed first-stage Krea 2 memory experiment: independent interventions."""
import os
import json

STEPS = 20
CHUNK_MEMORY_MIB = 192
CASES = (
    ('memory-fp32-control', 'fp32', 0),
    ('memory-bf16', 'bf16', 0),
    ('memory-int8-64', 'int8', 0),
    ('memory-splitmax-64', 'splitmax', 0),
    ('memory-activation-sync', 'fp32', 512),
)
PAIR_CASES = (
    ('pair-fp32-control', 'fp32', 0),
    ('pair-fp32-repeat', 'fp32', 0),
    ('pair-bf16', 'bf16', 0),
    ('pair-int8-64', 'int8', 0),
    ('pair-splitmax-64', 'splitmax', 0),
    ('pair-activation-sync', 'fp32', 512),
)
ASYNC_CASES = (PAIR_CASES[0], PAIR_CASES[1], PAIR_CASES[-1],
               ('pair-activation-async', 'fp32', 512))
PAIR_NAMES = {case for case, _, _ in PAIR_CASES + ASYNC_CASES}
FIXED_CASES = (
    ('fixed-control','fp32',0), ('fixed-repeat','fp32',0),
    ('fixed-sync','fp32',512), ('fixed-async','fp32',512),
)
FIXED_NAMES = {case for case, _, _ in FIXED_CASES}
CASE_MAP = {name:(storage, activation) for name,storage,activation in CASES + PAIR_CASES + ASYNC_CASES + FIXED_CASES}


def settings(group, output_root, common, paired=False, async_test=False, fixed_test=False):
    result=[]
    for case, storage, activation in (FIXED_CASES if fixed_test else ASYNC_CASES if async_test else PAIR_CASES if paired else CASES):
        args=' '.join(arg for arg in common['OPTIMIZER_ARGS'].split() if not arg.startswith('chunk_memory_mib='))
        args += f' chunk_memory_mib={CHUNK_MEMORY_MIB}'
        if storage!='fp32': args += f' momentum_storage={storage} momentum_block=64'
        result.append({**common,'__geometry_compare':case,'OPTIMIZER_TYPE':'muon_me',
                       'OPTIMIZER_ARGS':args,'LORA_NAME':f'{group}-{case}',
                       'LORA_OUTPUT_DIR':os.path.join(output_root,group,case),
                       'NETWORK_DIM':'64','NETWORK_ALPHA':'64','BLOCKS_SWAP':'22',
                       'DATASET_MEGAPIXELS':'0.25','MAX_TRAIN_EPOCHS':'1',
                       'COMPILE_BLOCKS':'Off','SAMPLE_ENABLED':False,'SAMPLE_AT_FIRST':False,
                       'SAVE_STATE':False,'SAVE_STATE_ON_TRAIN_END':False,
                       'VALIDATION_EVERY_STEPS':'0'})
    return result


def apply_flags(cmd,case):
    """Pin the same eager/checkpoint/streaming configuration in every arm."""
    _,activation=CASE_MAP[case]
    if '--batch_size_auto' in cmd:cmd.remove('--batch_size_auto')
    for flag in ('--compile_blocks','--gradient_checkpointing'):
        while flag in cmd:
            i=cmd.index(flag);del cmd[i:i+2]
    for flag in ('--send_only_swap','--no-send_only_swap','--send_only_force','--auto_reclaim_vram','--no-auto_reclaim_vram'):
        while flag in cmd:cmd.remove(flag)
    cmd.extend(['--compile_blocks','off','--gradient_checkpointing','on','--send_only_swap','--send_only_force',
                '--no-auto_reclaim_vram','--family_option',f'geometry_feasibility_steps={STEPS}',
                '--family_option','geometry_memory_phase=peaks','--family_option','geometry_memory_require_h2d=1',
                '--family_option',f'geometry_activation_mib={activation}','--family_option','geometry_host_floor_mib=6144'])
    if case in PAIR_NAMES:
        cmd.extend(['--family_option',f'geometry_memory_pair={case}'])
    if case in FIXED_NAMES:
        cmd.extend(['--family_option',f'geometry_fixed_probe={case}'])
    if case in ('pair-activation-async','fixed-async'):
        cmd.extend(['--family_option','geometry_activation_mode=async'])


def parse_memory_log(text):
    result={'memory_phases':[],'memory_states':[],'memory_config':[],'memory_losses':[],'pair_steps':[],'pair_complete':[],'fixed_trials':[],'fixed_complete':[]}
    for line in text.splitlines():
        for marker,key in (('[geometry-memory] ','memory_phases'),('[geometry-memory-state] ','memory_states'),('[geometry-memory-config] ','memory_config'),('[geometry-memory-loss] ','memory_losses'),('[geometry-pair-step] ','pair_steps'),('[geometry-pair-complete] ','pair_complete'),('[geometry-fixed-trial] ','fixed_trials'),('[geometry-fixed-complete] ','fixed_complete')):
            if marker not in line:continue
            try:
                value,_=json.JSONDecoder().raw_decode(line.split(marker,1)[1])
                if isinstance(value,dict):result[key].append(value)
            except (ValueError,TypeError):pass
    maxima={}
    for rec in result['memory_phases']:
        peak=rec.get('peak_bytes')
        if rec.get('ok') and isinstance(peak,(list,tuple)) and len(peak)==2:
            key=rec.get('phase');prior=maxima.get(key,[0,0]);maxima[key]=[max(a,b) for a,b in zip(prior,peak)]
    result['memory_phase_peaks_bytes']=maxima
    return result


def verify_memory_run(case,run):
    """A process exit code alone cannot establish that the intended arm ran."""
    storage,activation=CASE_MAP[case]
    config=run.get('memory_config',[]);phases=run.get('memory_phases',[]);states=run.get('memory_states',[])
    updates=(run.get('feasibility') or {}).get('updates',0)
    if not config or config[-1].get('offloader')!='LoRAStreamOffloader' or config[-1].get('activation_mib')!=activation:
        return 'Missing or mismatched memory configuration telemetry.'
    if case in FIXED_NAMES:
        return verify_fixed_run(case,run)
    if not updates or len(states)!=updates:return 'Incomplete memory-state telemetry.'
    if sorted(s.get('step') for s in states)!=list(range(1,updates+1)):
        return 'Unexpected memory-state steps.'
    for phase in ('forward','backward','optimizer'):
        rows=[r for r in phases if r.get('phase')==phase and r.get('ok') and r.get('peak_bytes') is not None]
        if len(rows)!=updates or sorted(r.get('step') for r in rows)!=list(range(1,updates+1)):
            return f'Incomplete {phase} phase telemetry.'
    if case in PAIR_NAMES:
        complete=run.get('pair_complete',[])
        traces=run.get('pair_steps',[])
        if len(complete)!=1 or complete[0].get('expected_steps')!=updates or [r.get('step') for r in traces]!=list(range(1,updates+1)):
            return 'Incomplete paired correctness trace.'
        meta=complete[0]
        if not meta.get('initial_sha256') or not meta.get('modules') or not meta.get('sample'):
            return 'Missing paired target manifest.'
        for r in traces:
            if not r.get('input_sha256') or (r['step'] in meta['horizons'] and set(r.get('metrics',{}))!=set(meta['sample'])):
                return 'Incomplete paired inputs or sampled metrics.'
    expected={'fp32':{'torch.float32'},'bf16':{'torch.bfloat16'},'int8':{'torch.float32','torch.int8'},'splitmax':{'torch.float32','torch.int8','torch.uint8'}}[storage]
    if set(states[-1].get('state_by_dtype',{}))!=expected or not states[-1].get('optimizer_state'):
        return 'Unexpected optimizer-state format.'
    if activation:
        last=states[-1].get('activation',{})
        if not last.get('accepted') or last.get('live')!=0 or last.get('peak',0)>last.get('cap',0):
            return 'No offloaded inputs or activation budget/lifetime failure.'
    if case == 'pair-activation-async':
        if config[-1].get('activation_mode')!='async' or (config[-1].get('activation_preflight') or {}).get('passed') is not True:
            return 'Missing CUDA async activation preflight.'
        for state in states:
            a=state.get('activation',{})
            if (a.get('live')!=0 or not a.get('accepted') or a.get('peak',0)>a.get('cap',0)
                or not a.get('d2h_bytes') or a.get('h2d_bytes',0)<a['d2h_bytes']
                or a.get('restores',0)<a['accepted']):
                return 'Async activation transfer accounting/lifetime mismatch.'
    return None


def verify_fixed_run(case,run):
    config=run.get('memory_config',[])
    fit=run.get('feasibility') or {}
    rows=run.get('fixed_trials',[])
    complete=run.get('fixed_complete',[])
    if not config or config[-1].get('fixed_case')!=case:
        return 'Missing fixed-state configuration.'
    mode='async' if case=='fixed-async' else 'sync' if case=='fixed-sync' else 'off'
    if config[-1].get('activation_mode')!=mode:
        return 'Unexpected fixed activation mode.'
    if mode=='async' and (config[-1].get('activation_preflight') or {}).get('passed') is not True:
        return 'Missing CUDA async activation preflight.'
    expected=[(s,t) for s in ('zero_up','nonzero_up') for t in (1,2,3)]
    if fit.get('updates')!=0 or fit.get('trials')!=6 or len(complete)!=1 or [ (r.get('state'),r.get('trial')) for r in rows]!=expected:
        return 'Incomplete fixed-state backward trials.'
    meta=complete[0]
    if meta.get('trials')!=6 or meta.get('optimizer_updates')!=0 or not meta.get('sample') or not meta.get('modules'):
        return 'Invalid fixed-state manifest.'
    inputs={r.get('input_sha256') for r in rows}
    if len(inputs)!=1 or not next(iter(inputs)):
        return 'Fixed input changed between trials.'
    for r in rows:
        if not r.get('factor_sha256') or r['factor_sha256']!=meta.get('states',{}).get(r['state']) or not r.get('gradient_sha256') or not r.get('output_sha256') or set(r.get('metrics',{}))!=set(meta['sample']):
            return 'Incomplete fixed-state evidence.'
        for boundary in ('forward_start','forward_end','backward_end'):
            if r.get(boundary,{}).get('optimizer_state_entries')!=0:
                return 'Unexpected optimizer state or missing boundary inventory.'
        if mode!='off':
            a=r.get('activation',{})
            if a.get('live')!=0 or not a.get('accepted') or a.get('peak',0)>a.get('cap',0):
                return 'Fixed activation lifetime/accounting mismatch.'
            if mode=='async' and (not a.get('d2h_bytes') or a.get('d2h_bytes')!=a.get('h2d_bytes') or a.get('restores')!=a.get('accepted')):
                return 'Fixed async transfer mismatch.'
    for phase in ('forward','backward'):
        actual=[r.get('step') for r in run.get('memory_phases',[]) if r.get('phase')==phase and r.get('ok') and r.get('peak_bytes')]
        if actual!=list(range(1,7)):
            return 'Incomplete fixed phase peaks.'
    return None
