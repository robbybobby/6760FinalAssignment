import os
import datetime
import numpy as np
import torch
from torch.optim import AdamW
import tqdm as tqdm
from rmu.utils import load_model, get_params, forward_with_cache, get_data

def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====RMU Config====")
    print("\n".join(f"{k}={v}" for k,v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    # Initialize control vectors for each forget dataset
    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(
            1, 1, updated_model.config.hidden_size, 
            dtype=updated_model.dtype, 
            device=updated_model.device
        )
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(args.num_epochs):
        print(f"======= Epoch {epoch} =======")
        epoch_unlearn_loss = 0
        epoch_retain_loss = 0
        
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss - push forget data activations toward control vector
                unlearn_inputs = tokenizer(
                    unlearn_batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=args.max_length
                ).to(updated_model.device)
                
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                )

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss - keep retain data activations similar to frozen model
                retain_inputs = tokenizer(
                    retain_batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=args.max_length
                ).to(updated_model.device)
                
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                )
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                )

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # Combined loss and optimization step
                total_loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                
                optimizer.step()

                epoch_unlearn_loss += unlearn_loss.item()
                epoch_retain_loss += retain_loss.item()

                # Logging
                if idx % args.log_interval == 0:
                    print(f"Batch {idx}: loss={total_loss.item():.4g} | "
                          f"unlearn={unlearn_loss.item():.4g} | "
                          f"retain={retain_loss.item():.4g}")
                
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
                    )
                    unlearn_cosine = torch.nn.functional.cosine_similarity(
                        updated_forget_activations, frozen_forget_activations, dim=-1
                    ).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(
                        updated_retain_activations, frozen_retain_activations, dim=-1
                    ).mean()
                    
                    print(f"  Forget cosine similarity: {unlearn_cosine.item():.4f}")
                    print(f"  Retain cosine similarity: {retain_cosine.item():.4f}")

                pbar.update(1)
        
        print(f"Epoch {epoch} - Avg unlearn loss: {epoch_unlearn_loss/num_batches:.4g}")
        print(f"Epoch {epoch} - Avg retain loss: {epoch_retain_loss/num_batches:.4g}")

    tokenizer.truncation_side = truncation_side
    
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{args.model_name_or_path.replace('/', '_')}_muse_rmu_{date}"
    
    os.makedirs(path, exist_ok=True)
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RMU for MUSE Dataset Unlearning")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, 
                       default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, 
                       default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Data arguments - Updated for MUSE
    parser.add_argument("--retain_corpora", type=str, 
                       default="retain_qa,retain_qa_icl",
                       help="MUSE retain datasets")
    parser.add_argument("--forget_corpora", type=str, 
                       default="forget_qa,forget_qa_icl", 
                       help="MUSE forget datasets")
    
    # RMU hyperparameters
    parser.add_argument("--alpha", type=str, default="50,50", 
                       help="Retain loss weights")
    parser.add_argument("--steering_coeffs", type=str, default="15,15",
                       help="Control vector magnitudes")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max sequence length for MUSE QA")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=10)
    
    # Architecture parameters
    parser.add_argument("--layer_id", type=int, default=15, 
                       help="Target layer for activation steering")
    parser.add_argument("--layer_ids", type=str, default="13,14,15", 
                       help="Layers to update")
    parser.add_argument("--param_ids", type=str, default="6", 
                       help="Parameter indices to update")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Parse string arguments
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    
    return args


if __name__ == "__main__":
    args = get_args()

    # Set seeds
    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load models and data
    print("Loading models...")
    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)
    
    print("Loading MUSE data...")
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        min_len=20,  # Adjusted for QA format
        max_len=args.max_length,
        batch_size=args.batch_size,
    )
    
    print(f"Loaded {len(forget_data_list)} forget datasets, {len(retain_data_list)} retain datasets")
    
    # Run RMU
    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )