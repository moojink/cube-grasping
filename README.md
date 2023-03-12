# Cube Grasping: Hand-Centric vs. Third-Person Perspectives

This repository includes source code for the first part of the paper titled [Vision-Based Manipulators Need to Also See from Their Hands](https://arxiv.org/abs/2203.12677). Please visit the [project website](https://sites.google.com/view/seeing-from-hands) for more information.

## Setup

1. Run the following commands to clone this repository, pull required submodules, create the `cube-grasping` conda environment, and install some packages inside the environment:

    ```
    git clone https://github.com/moojink/cube-grasping.git
    cd cube-grasping
    git submodule update --init --recursive
    conda create -n cube-grasping python=3.8
    conda activate cube-grasping
    ```


2. Run the following commands in the base directory of the repository to install the `cube-grasping`, `imitation`, and `stable-baselines3` packages into the conda environment. Note that we have made some modifications to these packages to fit our needs.

    ```
    pip install -e .
    pip install -e imitation
    pip install -e stable-baselines3
    ```

3. Install required packages via `pip`:

    ```
    pip install -r requirements.txt
    ```

4. Download the Describable Textures Dataset (DTD):

    ```
    wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    ```

5. Unzip the contents of `dtd-r1.0.1.tar.gz` in the base directory:

    ```
    tar -xzvf dtd-r1.0.1.tar.gz
    ```

    This should create a folder named `dtd` which contains a folder named `images`, which has many subfolders containing images of different texture styles.

## Running DAgger Experiments

Our [paper](https://arxiv.org/abs/2203.12677) experiments with three different distribution shifts: table height, distractor objects, and table texture. Please refer to the paper for more details.

### Running DAgger for Table Height Distribution Shift

1. Run the expert cube grasping demo collection script from the base directory:
    ```
    python cube_grasping/env/panda_expert.py --data_path=./data/panda_expert_demos_table_height.pkl --num_episodes=100
    ```
    This creates a pickle file, `panda_expert_demos.pkl`, containing 100 expert demos as well as two `MP4` files, `panda_expert_demos_view1.mp4` and `panda_expert_demos_view3.mp4`, which visualize the demos (where "`_view1`" and "`_view3`" correspond to the hand-centric and third-person perspectives, respectively).

2. Train a DAgger policy to grasp the target cube:
    ```
    python cube_grasping/scripts/train_dagger.py --data_path=./data/panda_expert_demos_table_height.pkl --log_dir=logs/dagger_table_height/ --num_dagger_rounds=6 --num_epochs_per_round=15 --num_trajectories_per_round=200 --test_type=table_height --seed=42
    ```
    Throughout training, the policy will be evaluated periodically (e.g. after every 5 epochs) on the training distribution and a set of out-of-distribution test distributions; we call the former "eval" and the latter "test." The results (e.g. mean rewards and success rates) will be printed out and also logged to `eval.csv` and several `test_{i}.csv` files in the log directory specified by `log_dir` above, where `i=0`, `i=1`, `i=2`, `i=3`, `i=4` correspond to table height shifts of -0.1m, -0.05m, +0m, +0.05m, +0.1m, respectively.

### Running DAgger for Distractor Objects Distribution Shift

The steps here are similar to the ones in the [table height section](#running-dagger-for-table-height-distribution-shift) above.

1. Run the expert cube grasping demo collection script from the base directory with the distractor objects flag enabled:
    ```
    python cube_grasping/env/panda_expert.py --data_path=./data/panda_expert_demos_distractors.pkl --num_episodes=100 --add_distractors=True

2. Train a DAgger policy to grasp the target cube:
    ```
    python cube_grasping/scripts/train_dagger.py --data_path=./data/panda_expert_demos_distractors.pkl --log_dir=logs/dagger_distractors/ --num_dagger_rounds=6 --num_epochs_per_round=15 --num_trajectories_per_round=200 --test_type=distractors --seed=42
    ```
    As before, eval/test results (e.g. mean rewards and success rates) will be printed out and also logged to `eval.csv` and several `test_{i}.csv` files in the log directory specified by `log_dir` above. Here, `i=0`, `i=1`, `i=2`, `i=3`, `i=4`, `i=5` correspond to distractor colors of red, green, blue, brown, white, black, respectively.

### Running DAgger for Table Textures Distribution Shift

The steps here are similar to the ones in the [table height section](#running-dagger-for-table-height-distribution-shift) and [distractor objects section](#running-dagger-for-distractor-objects-distribution-shift) above.

1. Run the expert cube grasping demo collection script from the base directory with the random table textures flag enabled:
    ```
    python cube_grasping/env/panda_expert.py --data_path=./data/panda_expert_demos_table_texture.pkl --num_episodes=100 --random_table_texture=True

2. Train a DAgger policy to grasp the target cube:
    ```
    python cube_grasping/scripts/train_dagger.py --data_path=./data/panda_expert_demos_table_texture.pkl --log_dir=logs/dagger_table_texture/ --num_dagger_rounds=6 --num_epochs_per_round=15 --num_trajectories_per_round=200 --test_type=distractors --seed=42
    ```
    As before, eval/test results (e.g. mean rewards and success rates) will be printed out and also logged to `eval.csv` and `test_0.csv` in the log directory specified by `log_dir` above.

## Running DrQ Experiments

This section discusses how to run the DrQ experiments for the three distribution shifts mentioned above.

### Running DrQ for Table Height Distribution Shift
Coming soon!

### Running DrQ for Distractor Objects Distribution Shift
Coming soon!


### Running DrQ for Table Textures Distribution Shift
Coming soon!
