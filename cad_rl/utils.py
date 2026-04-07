def _maybe_print_sample(completion, mesh_path, step, every=50):
    if every < 0 or step == 0 or step % every != 0:
        return
    print(
        f"\n[SAMPLE @ step {step}]\n Mesh path : {mesh_path} \n {completion}\n",
        flush=True,
    )
