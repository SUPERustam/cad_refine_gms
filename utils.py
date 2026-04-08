from logging_utils import log_event, truncate_text


def _maybe_print_sample(
    completion,
    mesh_path,
    step,
    every=50,
    logger=None,
    max_logged_completion_chars=4000,
    **fields,
):
    if every < 0 or step == 0 or step % every != 0:
        return
    if logger is None:
        return
    log_event(
        logger,
        "sample_payload",
        status="sampled",
        global_step=step,
        mesh_path=mesh_path,
        completion=truncate_text(completion, max_logged_completion_chars),
        **fields,
    )
