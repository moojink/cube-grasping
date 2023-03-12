"""Command-line arguments for ego/env/train_dagger.py."""

import argparse

def str_to_bool(s: str) -> bool:
    if s not in {'True', 'False'}:
        raise ValueError('Invalid boolean string argument given.')
    return s == 'True'

def get_args():
    parser = argparse.ArgumentParser(description='Process command-line args.')
    parser.add_argument("--data_path", type=str,
                        help="Path to the expert policy data used for training (required).")
    parser.add_argument("--use_view_1", type=str_to_bool, default=True,
                        help="Whether to train with the 1st-person POV camera.")
    parser.add_argument("--use_view_3", type=str_to_bool, default=True,
                        help="Whether to train with the 3rd-person POV camera.")
    parser.add_argument("--num_dagger_rounds", type=int, default=6,
                        help="Number of DAgger rounds to train for.")
    parser.add_argument("--num_epochs_per_round", type=int, default=15,
                        help="Number of epochs to train for per DAgger round.")
    parser.add_argument("--num_trajectories_per_round", type=int, default=200,
                        help="Number of trajectories to generate per DAgger round.")
    parser.add_argument("--log_dir", type=str, default='logs/dagger/',
                        help="Logs directory for TensorBoard stats and policy demo gifs.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0,
                        help="The epoch number from which to load BC checkpoint. If 0, start fresh.")
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory containing the saved BC checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the training environment.")
    parser.add_argument("--test_type", type=str,
                        help="'table_height', 'distractors', or 'table_texture'.")
    args = parser.parse_args()
    print("\nParsed the following command-line args:")
    for k, v in vars(args).items(): # vars(args) returns a dict version of args
        print('{}: {}'.format(k, v))
    print('') # newline
    # Ensure that the expert policy data path was given.
    if not args.data_path:
        raise argparse.ArgumentError("Missing required argument `--data`. Please give the full (absolute) path to the expert policy data.")
    # Add '/' to the end of args.log_dir if it's missing.
    if args.log_dir[-1] != '/':
        old_log_dir = args.log_dir
        args.log_dir += '/'
        print("Adding '/' to the end of args.log_dir.\nBefore: {}\nAfter: {}\n".format(old_log_dir, args.log_dir))
    # Verify that a `checkpoint_dir` arg was given if `checkpoint_epoch` > 0.
    if args.checkpoint_epoch > 0:
        assert args.checkpoint_dir is not None,\
            "If resuming training from checkpoint, you must provide a checkpoint directory (via --checkpoint_dir)."
    return args
