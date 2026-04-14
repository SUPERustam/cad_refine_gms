1. Use environment: `mamba activate cadtrl_restored`
2. To see logs from the vllm server: `tail -f logs/vllm_server.log`
3. To see logs from the training script: `tail -f logs/rl_base_train.log`
4. To see script: `tail -f train_loop_dp_rustam.sh`
5. To inspect CadQuery execution errors only: `rg "Error executing CadQuery code" logs/rl_base_train.log`
6. To inspect common root causes quickly: `rg "Error executing CadQuery code : ('result'|GC_MakeArcOfCircle|BRep_API|No pending wires|Null TopoDS_Shape|invalid decimal literal|name '.*' is not defined)" logs/rl_base_train.log`
7. Approx success rate from log (line-based): `python -c "import re; s=open('logs/rl_base_train.log',encoding='utf-8',errors='ignore').read(); steps=len(re.findall(r\"\\{'loss':\", s)); errs=len(re.findall(r\"Error executing CadQuery code\", s)); print({'steps_logged':steps,'cad_errors':errs,'approx_success_rate':None if steps==0 else max(0.0,1.0-errs/max(1,steps*16))})"`
8. Print generated sample code frequently (for debugging): set `--print_sample_steps 1` in `train_loop_dp_rustam.sh` (or larger value for less verbosity).