# Self-Rewarding-Language-Models

Paper implementation of [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020).

<img src="images/fig_1.png" width="700" height="300">

https://www.oxen.ai/datasets/Self-Rewarding-Language-Models/dir/main

sudo nvidia-smi -pl 300

2024-05-07 13:53:39,999 - root - ERROR - Error loading Model: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-05-07_13-43-07/dpo'. Use `repo_type` argument if needed.
2024-05-07 13:53:39,999 - root - ERROR - Error loading Model: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-05-07_13-43-07/dpo'. Use `repo_type` argument if needed.
2024-05-07 13:53:39,999 - root - ERROR - Error loading Model: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-05-07_13-43-07/dpo'. Use `repo_type` argument if needed.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/ds/workspace/Self-Rewarding-Language-Models/src/train/train.py", line 50, in <module>
    loader = ModelLoader(config, adapter=True, adapter_path=dpo_adapter_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ds/workspace/Self-Rewarding-Language-Models/src/utils/ModelLoader.py", line 28, in __init__
    self.model = self.load_model()
                 ^^^^^^^^^^^^^^^^^
  File "/home/ds/workspace/Self-Rewarding-Language-Models/src/utils/ModelLoader.py", line 64, in load_model
    lora_weights = load_peft_weights(self.adapter_path)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ds/anaconda3/envs/srlm/lib/python3.11/site-packages/peft/utils/save_and_load.py", line 297, in load_peft_weights
    has_remote_safetensors_file = file_exists(
                                  ^^^^^^^^^^^^
  File "/home/ds/anaconda3/envs/srlm/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 110, in _inner_fn
    validate_repo_id(arg_value)
  File "/home/ds/anaconda3/envs/srlm/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 158, in validate_repo_id
    raise HFValidationError(
huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-05-07_13-43-07/dpo'. Use `repo_type` argument if needed.