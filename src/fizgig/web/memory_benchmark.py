"""Fixed first-stage Krea 2 memory experiment: independent interventions."""
import os
import json

STEPS = 20
CASES = (
    ('memory-fp32-control', 'fp32', 0),
    ('memory-bf16', 'bf16', 0),
    ('memory-int8-64', 'int8', 0),
    ('memory-splitmax-64', 'splitmax', 0),
    ('memory-activation-sync', 'fp32', 512),
)
CASE_MAP = {name:(storage, activation) for name,storage,activation in CASES}


def settings(group, output_root, common):
    result=[]
    for case, storage, activation in CASES:
        args=common['OPTIMIZER_ARGS']
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


def parse_memory_log(text):
    result={'memory_phases':[],'memory_states':[],'memory_config':[]}
    for line in text.splitlines():
        for marker,key in (('[geometry-memory] ','memory_phases'),('[geometry-memory-state] ','memory_states'),('[geometry-memory-config] ','memory_config')):
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
    if not updates or len(states)!=updates:return 'Incomplete memory-state telemetry.'
    if sorted(s.get('step') for s in states)!=list(range(1,updates+1)):
        return 'Unexpected memory-state steps.'
    for phase in ('forward','backward','optimizer'):
        rows=[r for r in phases if r.get('phase')==phase and r.get('ok') and r.get('peak_bytes') is not None]
        if len(rows)!=updates or sorted(r.get('step') for r in rows)!=list(range(1,updates+1)):
            return f'Incomplete {phase} phase telemetry.'
    expected={'fp32':{'torch.float32'},'bf16':{'torch.bfloat16'},'int8':{'torch.float32','torch.int8'},'splitmax':{'torch.float32','torch.int8','torch.uint8'}}[storage]
    if set(states[-1].get('state_by_dtype',{}))!=expected or not states[-1].get('optimizer_state'):
        return 'Unexpected optimizer-state format.'
    if activation:
        last=states[-1].get('activation',{})
        if not last.get('accepted') or last.get('live')!=0 or last.get('peak',0)>last.get('cap',0):
            return 'No offloaded inputs or activation budget/lifetime failure.'
    return None
