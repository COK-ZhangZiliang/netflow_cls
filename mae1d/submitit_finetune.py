import argparse
import os
import uuid
from pathlib import Path

import main_finetune as classification
import submitit


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE finetune", parents=[classification_parser])
    
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--dataset", default="DoH", type=str, help="Dataset to use")
    parser.add_argument("--sample_rate", default="60", type=str, help="Sample rate to use")
    
    parser.add_argument("--ablation", default=False, action="store_true", help="Abalation study")

    return parser.parse_args()


def get_shared_folder(args) -> Path:
    if args.ablation:
        if args.frozen_embed:
            p = Path(f"./checkpoint/ablation/frozen_embed/finetune/{args.dataset}/{args.sample_rate}")
        else:
            p = Path(f"./checkpoint/ablation/path_size/{args.patch_size}/finetune/{args.dataset}/{args.sample_rate}")
    else:
        p = Path(f"./checkpoint/finetune/{args.dataset}/{args.sample_rate}")
    os.makedirs(p, exist_ok=True)
    return p.absolute()


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args, rate, dataset):
        self.args = args
        self.rate = rate
        self.dataset = dataset

    def __call__(self):
        import main_finetune as classification

        self._setup_gpu_args()
        classification.main(self.args, self.rate, self.dataset)


    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=40,
        gpus_per_node=2,
        tasks_per_node=1, # one task per GPU
        cpus_per_task=10,
        nodes=1,
        timeout_min=timeout_min,  # max is 60 * 72
    )

    executor.update_parameters(name="mae")

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    sample_rate = [int(rate) for rate in args.sample_rate.split(",")]
    dataset = args.dataset
    for rate in sample_rate:
        trainer = Trainer(args, rate, dataset)
        job = executor.submit(trainer)
        
        print(job.job_id)


if __name__ == "__main__":
    main()
