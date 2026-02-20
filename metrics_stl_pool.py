import cadquery as cq
import os
import numpy as np

def init_worker():
    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['cq'] = cq


class Wrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, args, shared_args_idx, results_q=None):
        while shared_args_idx.value < len(args):
            res = self.func(*args[shared_args_idx.value])
            if results_q is not None:
                results_q.put(res)
            shared_args_idx.value += 1


class ProcessPool:
    def __init__(self, task_func, task_args: list[tuple], n_processes: int = 16, timeout: float = 5):
        self.n_processes = n_processes
        self.timeout = timeout
        self.task_func = task_func
        self.task_args = task_args

    def run(self):
        import time
        from multiprocessing import Process, Value, Queue, get_context

        CTX = get_context("spawn")

        from queue import Empty

        from tqdm import tqdm

        pbar = tqdm(total=len(self.task_args))
        task_args = self.split_list(self.task_args, self.n_processes)
        shared_args_indicies = [CTX.Value('i', 0) for _ in range(self.n_processes)]
        last_args_indicies = [0 for _ in range(self.n_processes)]
        results_q = CTX.Queue()
        results = []
        pool = [CTX.Process(target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i], results_q), daemon=True)
                for i in range(self.n_processes)]
        unprocessed_args = []

        for process in pool:
            process.start()

        n_processes_running = self.n_processes

        while n_processes_running:
            time.sleep(self.timeout)
            for i in range(self.n_processes):
                if pool[i] is None:
                    continue

                if shared_args_indicies[i].value >= len(task_args[i]):
                    process = pool[i]
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=30)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    pool[i] = None
                    n_processes_running -= 1
                elif shared_args_indicies[i].value == last_args_indicies[i]:
                    hang_process = pool[i]
                    hang_process.terminate()
                    hang_process.join(timeout=30)
                    if hang_process.is_alive():
                        hang_process.kill()
                        hang_process.join()

                    unprocessed_args.append(task_args[i][shared_args_indicies[i].value])
                    #with shared_args_indicies[i].get_lock():
                    shared_args_indicies[i].value += 1
                    new_process = CTX.Process(
                        target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i], results_q), daemon=True
                    )
                    new_process.start()
                    pool[i] = new_process

                last_args_indicies[i] = shared_args_indicies[i].value

            pbar.update(sum(last_args_indicies) - pbar.n)
            while True:
                try:
                    results.append(results_q.get_nowait())
                except Empty:
                    break

        pbar.close()
        while True:
            try:
                results.append(results_q.get_nowait())
            except Empty:
                break
        results_q.close()
        results_q.join_thread()

        return results, unprocessed_args

    def split_list(self, lst, num_chunks):
        n = len(lst)
        base_size, remainder = divmod(n, num_chunks)
        sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

        chunks = []
        start = 0
        for size in sizes:
            chunks.append(lst[start: start + size])
            start += size
        return chunks


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces, process=True)


def compute_iou(pred_mesh, gt_mesh):
    intersection_volume = 0
    for gt_mesh_i in gt_mesh.split():
        for pred_mesh_i in pred_mesh.split():
            intersection = gt_mesh_i.intersection(pred_mesh_i)
            volume = intersection.volume if intersection is not None else 0
            intersection_volume += volume

    gt_volume = sum(m.volume for m in gt_mesh.split())
    pred_volume = sum(m.volume for m in pred_mesh.split())
    union_volume = gt_volume + pred_volume - intersection_volume
    iou = intersection_volume / (union_volume + 1e-6)
    return iou


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def transform_mesh_0_1(mesh):
    # scale a mesh to be centered and inside [0,1]
    if mesh is None:
        print(f"mesh is none")
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def compute_metrics(py_text, gt_mesh_path, n_points, var_name="result", normalize="fixed"):
    init_worker()
    base_file = os.path.basename(gt_mesh_path).rsplit('.stl', 1)[0]
    pred_mesh = py_file_to_mesh_file(py_text, var_name)
    if pred_mesh == None : 
        print(f"no path")
        return dict(file_name=base_file, cd=None, iou=None, auc=None)
    cd, iou, auc = None, None, None
    try: 
        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        if normalize == "fixed":
            gt_mesh = transform_mesh_0_1(gt_mesh)
            pred_mesh = transform_pred_mesh(pred_mesh)
        if normalize == "mesh_extents":
            gt_mesh = transform_mesh_0_1(gt_mesh)
            pred_mesh = transform_mesh_0_1(pred_mesh)
        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        try:
            iou = compute_iou(gt_mesh, pred_mesh)
        except Exception as e:
            print(f"exception during execution {e}")
            iou = None
            pass
        print(f"cd : {cd}, iou {iou}")
    except Exception as e:
        print(f"exception during execution {e}")
        pass
    
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc)

    


def py_file_to_mesh_file(py_text, var_name):
    try:
        namespace = {"cq": cq}
        exec(py_text, namespace)
        compound = namespace[var_name].val()
        assert len(compound.Faces()) > 2
        return compound_to_mesh(compound)
    except Exception as ex:
        print(f"Exception {ex}")
        return None
    


def run_texts(py_texts, gt_paths, var_name="result", normalize="fixed", n_processes=16):
    n_points = 8192
    n = len(gt_paths)

    pool = ProcessPool(task_func=compute_metrics, timeout = 16, task_args=list(zip(py_texts, gt_paths, [n_points] * n, [var_name]*n, [normalize]*n)))
    results, unprocessed_args = pool.run()
    print(f"unprocessed args {len(unprocessed_args)}")
    for _ in unprocessed_args:
        results.append(None)
    

    return results