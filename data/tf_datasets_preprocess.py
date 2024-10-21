import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import imageio
import os
import argparse
import json

def main():
    builder = tfds.builder_from_directory(builder_dir=args.tf_dataset_path)
    ds = builder.as_dataset(split='train')

    os.makedirs(f'./{args.dataset_name}', exist_ok=True)

    json_list = []

    for episode_id, episode in enumerate(iter(ds)):
        
        os.makedirs(f'./{args.dataset_name}/episode_{episode_id}')



        steps = list(episode['steps'])

        language_instructions = [step['language_instruction'].numpy().decode() for step in steps]
        images = [cv2.resize(np.array(step['observation']['image']), (224, 224)) for step in steps]

        for step_id in range(len(images)):
            cv2.imwrite(f'./{args.dataset_name}/episode_{episode_id}/step_{step_id}.jpg', cv2.cvtColor(images[step_id], cv2.COLOR_RGB2BGR))
            
            raw_actions = steps[step_id]['action'].numpy().tolist()

            json_item =  {
            "id": f"data/finetune_data/episode_{episode_id}/step_{step_id}",
            "image": f"data/finetune_data/episode_{episode_id}/step_{step_id}.jpg",
            "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n"+language_instructions[step_id]
                    },
                    {
                        "from": "gpt",
                        "raw_actions": raw_actions
                        
                    }
                ]
            }
            json_list.append(json_item)
    
    with open('openvla_beef_test.json', 'w') as file:
        json.dump(json_list, file)


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser('convert tensorflow dataset to the format for preprocessing')

    argparser.add_argument('--dataset_name', type=str, default='finetune_data')
    argparser.add_argument('--tf_dataset_path', type=str)
    

    args = argparser.parse_args()
    main()