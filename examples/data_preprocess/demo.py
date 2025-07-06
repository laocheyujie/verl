import os
from pathlib import Path

import datasets


def main(data_files, save_dir):
    """
    Preprocess json dataset to parquet format
    """
    data_path = os.path.splitext(Path(data_files[0]))[-1][1:]
    raw_data = datasets.load_dataset(data_path, data_files=data_files, num_proc=16, streaming=False)

    raw_test_data = raw_data["train"].select(range(200))
    raw_train_data = raw_data["train"].select(range(200, len(raw_data["train"])))
    print(f"train_data size: {len(raw_train_data)}, test_data size: {len(raw_test_data)}")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("instruction")
            example.pop("input")
            example.pop("output")
            
            data = {
                "data_source": "aminer",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "instruction",
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "split": split,
                    "index": idx
                },
            }
            return data

        return process_fn
    
    train_data = raw_train_data.map(function=make_map_fn("train"), with_indices=True)
    test_data = raw_test_data.map(function=make_map_fn("test"), with_indices=True)
    
    train_data.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_data.to_parquet(os.path.join(save_dir, "test.parquet"))
    
    print("Done!")


if __name__ == "__main__":
    data_files = ["/data/cheyujie/github_fork/verl/data/all_ppo.json"]
    save_dir = "/data/cheyujie/github_fork/verl/data/aminer"
    main(data_files, save_dir)