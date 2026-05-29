import json
import sys
import os
import argparse
from openai import OpenAI
from openai import AzureOpenAI
from inference_engine.vis_inference_demo_gpt import evaluate_single_data, evaluate_single_with_cleanup
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor, ImageRuntime, DistFittingRuntime

from plotting_utils import *
import pickle
import pandas as pd
import numpy as np
from anthropic import AnthropicBedrock
import re

def clean_data_name(s):
    match = re.search(r'_(\w+)\.', s)
    return match.group(1) if match else None

def main():
    """Main function with command-line arguments support"""
    parser = argparse.ArgumentParser(description='Visual Question Answering with Code Execution')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default="./test_data/one_image_demo.png",
                        help='Path to the input image')
    parser.add_argument('--question', type=str, 
                        default="From the information on that advertising board, what is the type of this shop?",
                        help='Question to ask about the image')
    
    # Configuration arguments
    parser.add_argument('--api_config', type=str, default="./api_config.json",
                        help='Path to API configuration file')
    parser.add_argument('--client_type', type=str, default="openai",
                        help='Client Type')
    parser.add_argument('--prompt_template', type=str, default="./prompt_template/prompt_template_vis.json",
                        help='Path to prompt template file')
    parser.add_argument('--prompt', type=str, default="vistool_with_img_info_v2",
                        help='Prompt type to use')
    
    # Execution arguments
    parser.add_argument('--exe_code', action='store_true', default=True,
                        help='Whether to execute code blocks')
    parser.add_argument('--max_tokens', type=int, default=10000,
                        help='Maximum tokens for response')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for generation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="./test_data",
                        help='Directory to save output files')
    parser.add_argument('--save_messages', action='store_true', default=True,
                        help='Whether to save the message history')
    # NEW   
    parser.add_argument('--max_code_steps', type=int, default=5,
                        help='MAX number of coding steps pyvision can take')
    parser.add_argument('--dataset', type=str, default=5,
                        help='Desides which dataset to run') 
    args = parser.parse_args()
     
    # Load API configuration
    with open(args.api_config, 'r') as f:
        api_config = json.load(f)

    with open(args.dataset, 'rb') as f:
        dataset= pickle.load(f)
        # print("WARNING: RUNNING ON MINI DATASET")
        # dataset = dataset[:2]  # Limit to 2 samples for testing.

    # Detect task type from prompt template path
    is_ts = 'prompt_template_ts' in args.prompt_template
    os.makedirs(args.output_dir, exist_ok=True) 

    results_df = []
    #print("RUNNING ON LAST 50")
    # TODO REMOVE FROM SAVE PATH ONCE FINISHED
    #dataset=dataset[50:]

    for idx, raw_data in enumerate(dataset):
        # Plot and set image path based on task type
        if is_ts:
            # TODO IMPLEMENT TS 
            plot_time_series(raw_data['data'], save_path='timeseries.png')
            image_path = os.path.join(os.path.dirname(args.dataset), 'timeseries.png') if not os.path.isabs('timeseries.png') else 'timeseries.png'
            image_path = 'timeseries.png'
            sample_data = np.array(raw_data['data'].values, dtype=np.float64)
        else:
            image_path = args.image_path
            plot_hist(raw_data['data'], save_path=image_path) 
            sample_data = raw_data['data']

        # Initialize client based on client_type
        if args.client_type == "azure": 
            print("API CONFIG", api_config) 
            client = AzureOpenAI(
                api_key=api_config['azure_openai_api_key'][0],
                azure_endpoint=api_config['azure_openai_endpoint'],
                api_version=api_config.get('api_version', '2024-02-01')
            )
        elif args.client_type == "bedrock":
            client = AnthropicBedrock(
                aws_access_key=api_config["aws_access_key"][0],
                aws_secret_key=api_config["aws_secret_key"][0],
                aws_region=api_config["aws_region"],
            )
        else:
            api_key = api_config['api_key'][0]
            base_url = api_config.get('base_url', None)
            client = OpenAI(api_key=api_key, base_url=base_url)
            
        # Prepare data
        data = {
            "question": args.question,
            "image_path_list": [image_path],
        }
        
        # Prepare arguments
        eval_args = {
            "max_tokens": args.max_tokens,
            "prompt_template": args.prompt_template,
            "prompt": args.prompt,
            "exe_code": args.exe_code,
            "temperature": args.temperature,
            "client_type": args.client_type,
            "api_name": api_config['model'],
            "max_code_steps": args.max_code_steps,
            "idx": idx
        }
        
        # Run inference with safe execution
        print(f"Processing image: {image_path}")
        print(f"Question: {args.question}")
        print(f"Task type: {'time_series' if is_ts else 'distribution_fitting'}")
        print("Running inference with safe execution...")
    
        executor = PythonExecutor(runtime_class=DistFittingRuntime, init_vars={'DATA': sample_data})
        messages, final_response = evaluate_single_data(eval_args, data, client, executor, args.output_dir)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.save_messages:
            messages_path = os.path.join(args.output_dir, f"test_messages_{idx}.json")
            with open(messages_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=4, ensure_ascii=False)
            print(f"Messages saved to: {messages_path}")
        
        # Print response
        print("\n" + "="*50)
        print("Final Response:")
        print("="*50)
        print(final_response)
        
        tools_path = f"tools_{idx}.json"
        with open(os.path.join(args.output_dir, tools_path), "r", encoding="utf-8") as f:
            tool_calls = json.load(f)  # loads as a list of strings

        results_df.append({
            "idx": idx,
            "final_response": final_response,
            "tool_calls": tool_calls  # pandas will str()-ify the list
        })

    pd.DataFrame(results_df).to_csv(f'pyvision_{args.output_dir}_results.csv', index=False)


if __name__ == "__main__":
    main()