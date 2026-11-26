from types import SimpleNamespace
from fizgig.web import memory_benchmark as m

def test_matched_settings_and_flags(tmp_path):
    c=m.settings('test',str(tmp_path),{'OPTIMIZER_ARGS':'chunk_memory_mib=128','SEED':'42','VALIDATION_HOLDOUT':True})
    assert len(c)==5 and len({v['LORA_OUTPUT_DIR'] for v in c})==5
    exclude={'__geometry_compare','LORA_NAME','LORA_OUTPUT_DIR','OPTIMIZER_ARGS'}
    assert all({k:v for k,v in x.items() if k not in exclude}=={k:v for k,v in c[0].items() if k not in exclude} for x in c)
    for x in c:
        cmd=['python','--compile_blocks','auto','--no-send_only_swap','--gradient_checkpointing','auto','--batch_size_auto']
        m.apply_flags(cmd,x['__geometry_compare'])
        assert cmd.count('--compile_blocks')==1 and cmd[cmd.index('--compile_blocks')+1]=='off'
        assert '--no-send_only_swap' not in cmd and '--batch_size_auto' not in cmd and '--send_only_force' in cmd
        assert 'geometry_feasibility_steps=20' in cmd and 'geometry_memory_phase=peaks' in cmd
        assert ('geometry_activation_mib=512' in cmd)==(x['__geometry_compare']=='memory-activation-sync')

def test_report_requires_actual_intervention():
    raw='[geometry-memory-config] {"offloader":"LoRAStreamOffloader","activation_mib":512}\n'
    raw+='[geometry-memory-state] {"step":1,"optimizer_state":100,"state_by_dtype":{"torch.float32":100},"activation":{"accepted":3,"live":0,"peak":512,"cap":512}}\n'
    for phase in ('forward','backward','optimizer'):
        raw+='[geometry-memory] {"step":1,"phase":"'+phase+'","ok":true,"peak_bytes":[10,20]}\n'
    r=m.parse_memory_log(raw);r['feasibility']={'updates':1}
    assert m.verify_memory_run('memory-activation-sync',r) is None
    assert m.verify_memory_run('memory-bf16',r)
    r['memory_states'][0]['activation']['accepted']=0
    assert m.verify_memory_run('memory-activation-sync',r)
    assert r['memory_phase_peaks_bytes']['backward']==[10,20]
