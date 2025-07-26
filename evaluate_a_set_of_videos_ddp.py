import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import pickle as pkl

import decord
import numpy as np
import yaml
from tqdm import tqdm

from dover.datasets import (
    UnifiedFrameSampler,
    ViewDecompositionDataset,
    spatial_temporal_view_decomposition,
)
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    ## results[0]: aesthetic, results[1]: technical
    ## thank @dknyxh for raising the issue
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


def setup_ddp():
    """Initialize DDP for single machine"""
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    
    return rank, world_size


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--opt", type=str, default="./dover.yml", help="the option file"
    )

    parser.add_argument(
        "-in",
        "--input_video_dir",
        type=str,
        default="./demo",
        help="the input video dir",
    )

    parser.add_argument(
        "-out",
        "--output_result_csv",
        type=str,
        default="./dover_predictions/demo.csv",
        help="the output result csv file",
    )

    parser.add_argument(
        "-bs", "--batch_size", type=int, default=4, help="batch size for evaluation"
    )

    args = parser.parse_args()

    # Setup DDP
    rank, world_size = setup_ddp()
    
    # Set device
    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load DOVER
    evaluator = DOVER(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=device)
    )

    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        evaluator = DDP(evaluator, device_ids=[rank])

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_result_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Only rank 0 writes the header
    if rank == 0:
        with open(args.output_result_csv, "w") as w:
            w.write(f"path, aesthetic score, technical score, overall/final score\n")

    dopt = opt["data"]["val-l1080p"]["args"]
    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    dataset = ViewDecompositionDataset(dopt)

    # Create distributed sampler if using multiple GPUs
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            num_workers=opt["num_workers"], 
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=opt["num_workers"], 
            pin_memory=True,
            shuffle=False,
        )

    sample_types = ["aesthetic", "technical"]
    all_results = {}

    # Set model to evaluation mode
    evaluator.eval()

    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Testing on GPU {rank}")
    else:
        pbar = dataloader

    with torch.no_grad():
        for i, data in enumerate(pbar):
            if len(data.keys()) == 1:
                # failed data
                continue

            batch_size = len(data["name"])

            # Process each video in the batch
            for batch_idx in range(batch_size):
                video = {}
                for key in sample_types:
                    if key in data:
                        # Extract single video from batch
                        single_video_data = data[key][batch_idx:batch_idx+1].to(device)
                        b, c, t, h, w = single_video_data.shape
                        video[key] = (
                            single_video_data
                            .reshape(
                                b, c, data["num_clips"][key][batch_idx], t // data["num_clips"][key][batch_idx], h, w
                            )
                            .permute(0, 2, 1, 3, 4, 5)
                            .reshape(
                                b * data["num_clips"][key][batch_idx], c, t // data["num_clips"][key][batch_idx], h, w
                            )
                        )

                # Get model predictions
                if world_size > 1:
                    results = evaluator.module(video, reduce_scores=False)
                else:
                    results = evaluator(video, reduce_scores=False)
                
                results = [np.mean(l.cpu().numpy()) for l in results]
                rescaled_results = fuse_results(results)
                
                video_name = data["name"][batch_idx]
                all_results[video_name] = rescaled_results

    # Gather results from all processes if using DDP
    if world_size > 1:
        # Gather all results to rank 0
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, all_results)
        
        if rank == 0:
            # Merge all results
            final_results = {}
            for results_dict in gathered_results:
                if results_dict is not None:
                    final_results.update(results_dict)
            all_results = final_results

    # Only rank 0 writes the final results
    if rank == 0:
        print(f"Processed {len(all_results)} videos in total.")
        
        # Write results to CSV
        with open(args.output_result_csv, "a") as w:
            for video_name, rescaled_results in all_results.items():
                w.write(
                    f'{video_name}, {rescaled_results["aesthetic"]*100:.4f}, {rescaled_results["technical"]*100:.4f}, {rescaled_results["overall"]*100:.4f}\n'
                )

        # Write results to text file (similar to original code)
        output_txt = f"zero_shot_res_{args.input_video_dir.split('/')[-1]}.txt"
        with open(output_txt, "w") as wf:
            for video_name, rescaled_results in all_results.items():
                video_filename = video_name.split("/")[-1]
                wf.write(f'{video_filename},{rescaled_results["aesthetic"]*100:.4f}, {rescaled_results["technical"]*100:.4f}, {rescaled_results["overall"]*100:.4f}\n')

        print(f"Results saved to {args.output_result_csv} and {output_txt}")

    # Clean up DDP
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 