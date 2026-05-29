#!/bin/bash


# # openai client

# python main.py \
#     --image_path ./test_data/one_image_demo.png \
#     --question "What is the color of the liquid contained in the glass on the table?" \
#     --api_config ./api_config_files/api_config_openai.json \
#     --client_type openai \
#     --prompt_template ./prompt_template/prompt_template_vis.json \
#     --prompt vistool_with_img_info_v2 \
#     --exe_code \
#     --max_tokens 10000 \
#     --temperature 0.6 \
#     --output_dir ./test_data \
#     --save_messages 

# azure client

# python main.py \
#     --image_path ./test_data/one_image_demo.png \
#     --question What is the color of the liquid contained in the glass on the table? \
#     --api_config ./api_config_files/api_config_azure.json \
#     --client_type azure \
#     --prompt_template ./prompt_template/prompt_template_vis.json \
#     --prompt vistool_with_img_info_v2 \
#     --exe_code \
#     --max_tokens 10000 \
#     --temperature 0.6 \
#     --output_dir ./test_data \
#     --save_messages 

# vllm client 

# python main.py \
#     --image_path ./test_data/one_image_demo.png \
#     --question What is the color of the liquid contained in the glass on the table? \
#     --api_config ./api_config_files/api_config_vllm.json \
#     --client_type vllm \
#     --prompt_template ./prompt_template/prompt_template_vis.json \
#     --prompt vistool_with_img_info_v2 \
#     --exe_code \
#     --max_tokens 10000 \
#     --temperature 0.6 \
#     --output_dir ./test_data \
#     --save_messages 

IMAGE_PATH="./fit.png"
QUESTION="Propose a PyMC model that best fits the data."
API_CONFIG="./api_config_files/api_config_bedrock.json"  #"./api_config_files/api_config_azure.json"
CLIENT_TYPE="bedrock"
PROMPT_TEMPLATE="./prompt_template/prompt_template_dist.json"
PROMPT="pyvision_fitting"
MAX_TOKENS=4028
TEMPERATURE=0.6
OUTPUT_DIR="claude_output"
MAX_CODE_STEPS=5
DATASET='data_mixed.pkl'
# --- Run ---
python main.py \
    --image_path "$IMAGE_PATH" \
    --question "$QUESTION" \
    --api_config "$API_CONFIG" \
    --client_type "$CLIENT_TYPE" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --prompt "$PROMPT" \
    --exe_code \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --max_code_steps "$MAX_CODE_STEPS" \
    --dataset "$DATASET" \
    --save_messages