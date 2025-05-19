# bocconi-nlrl

Implementing [Natural Language Reinforcement Learning](https://arxiv.org/pdf/2411.14251).

### Run
* Setup
  * Dependencies: `conda env create -f environment.yml` 
* Maze
  * Train: `python train_maze.py configs/maze/train.json`
  * Evaluate: `python eval_maze.py configs/maze/eval.json`
* Negotiation
  * Train: `python train_nego.py configs/negotiation/train.json`
* Running on slurm
  * `sbatch --account={id} create_env.sh`
  * `sbatch --account={id} train_maze.sh`

### Repository layout

`/bocconi-nlrl` - repository directory

* `/agents` - directory containing the agent and its core components.
  * `actorcritic.py` - code for training the actor-critic agent.
  * `improvement.py` - code for the improvement operator.
  * `language_policy.py` - code for the language policy.
  * `language_value_function.py` - code for the language value function.
 
* `configs` - directory containing all the hyperparameters and file locations for each object
  * `agent.json` - parameters for the actor-critic agent.
  * `improvement.json` - parameters for the improvement operator.
  * `mistral.json` - parameters for the Mistral LLM.
  * `policy.json` - parameters for the language policy.
  * `values.json` - parameters for the language value function.

* `\envs` - directory containing the different environments
  * `environment.py` - abstract environment base class
  * `\maze` - directory for the maze environment
     * `maze.py` - code defining the maze game.
     * `\starting_boards` - directory containing the text representation of mazes
         * `simple_board.txt` - 3x3 maze. 

 * `\models` - directory containing the language models
    * `model.py` - abstract LLM base class.
    * `mistral.py` - Mistral LLM class.

 * `\prompts` - directory containing the LLM prompts, stored as text files.
    * `\policy` - directory containing prompts for the language policy.
      * `system_prompt.txt`
      * `user_prompt.txt`
    * `value` - directory containing prompts for the language value function.
      * `describe_transition.txt`
      * `mc_estimate_prompt.txt`
      * `system_prompt.txt`
      * `value_prompt.txt`

 * `train_maze.py` - high level function that starts the training procedure.

### Pre-NLRL training finetuning of Mistral

The partially fine-tuned Mistral model is too large for GitHub. It is uploaded on Google Drive:
https://drive.google.com/drive/u/0/folders/14LJ-j1fb3MNa6gUlvGfiFeecZzgApmWi
