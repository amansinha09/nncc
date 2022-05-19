# nncc
Code repository for 2022 n2c2 Shared Task

Initial Report: https://docs.google.com/document/d/1eIhZYO8q-OZMcaEtnpGqPmnt3C1BTi1dz6OHnGB2I40/edit

Initial Notebook: https://www.kaggle.com/code/ayanvishwakarma/notebook13e5fae30b

Final Report: https://docs.google.com/document/d/1zYaTDeaDgR10HpIvybmVQ6HMpVKvPYhOtQwMQAcnrH4/edit

Final Notebook: https://www.kaggle.com/code/ayanvish/notebook32f62f9e14

**Training**

Please make sure that the Word Embeddings are downloaded (manually) in the Pretrained Word Embeddings directory.

```python

python PATH_TO_training.py --train_dir PATH_TO_TRAIN_FOLDER --valid_dir  PATH_TO_DEV_FOLDER --output_folder  PATH_TO_SYSTEM_OUTPUT_FOLDER --save_path PATH_TO_save_model.pt --head [linear|linear_lstm] --on_task [task_1|task_2]

```
