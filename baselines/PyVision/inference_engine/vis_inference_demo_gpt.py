import sys
import os
import re
import json
import base64
from io import BytesIO
from tracemalloc import stop
from PIL import Image
import argparse
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor, ImageRuntime
from openai import OpenAI
import anthropic
 
def encode_image(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, and get resolution info.
    
    Args:
        image: Can be a PIL.Image object or image file path.
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None
    
    if isinstance(image, str):
        # Handle file path
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL.Image object
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }
 
def encode_image_with_resize(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, get resolution info.
    If resolution > 1024x1024, resize to half.
    
    Args:
        image: Can be a PIL.Image object or image file path
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None
    
    if isinstance(image, str):
        img_obj = Image.open(image)
    else:
        img_obj = image
 
    # Resize if larger than 1024x1024
    width, height = img_obj.size
    if width > 1024 or height > 1024:
        new_size = (width // 2, height // 2)
        img_obj = img_obj.resize(new_size, Image.LANCZOS)
        width, height = img_obj.size
 
    buffered = BytesIO()
    img_obj.save(buffered, format='PNG')
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
 
    return {
        'base64': base64_str,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}"
    }
 
def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness
 
def execute_codes(codes, messages, executor: PythonExecutor):
    no_code_idx = []
    codes_use = []
    for i, code in enumerate(codes):
        if code == "":
            no_code_idx.append(i)
        else:
            codes_use.append(code)
    batch_results = executor.batch_apply(codes_use, messages)
    return batch_results, no_code_idx
 
def process_prompt_init(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]
 
    image_path = image_path_list[0]
 
    if "<IMAGE_PLACE_HOLDER_0>" in question:
        if "no_tool" in prompt_type:
 
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question
            question = prompt_prefix.format(query=question_with_options)
 
            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            content = []
            
            # Add text before image (if any)
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            # Add image
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            
            # Add text after image (if any)
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})
 
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
 
            return messages
 
        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question
            question = prompt_prefix.format(query=question_with_options, width=str(width), height=str(height))
 
            # Split question into parts
            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            # Build message with image_clue tags
            content = []
            
            # Add text before image (if any)
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            
            # Add image with tags
            content.append({"type": "text", "text": "<image_clue_0>"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            content.append({"type": "text", "text": "</image_clue_0>\n\n"})
            
            # Add text after image (if any)
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})
 
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
 
            return messages
 
    else:
        if "no_tool" in prompt_type:
 
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question
 
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options)}]
                }
            ]
 
            return messages
 
        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question
 
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<image_clue_0>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": "</image_clue_0>\n\n"}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options, width=str(width), height=str(height))}]
                }
            ]
 
            return messages
 
def process_prompt_init_multi_images(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]
    
    # Prepare image data
    image_data = []
    image_information = ""
    
    for i, image_path in enumerate(image_path_list):
        if "claude" in api_name:
            img_result = encode_image_with_resize(image_path)
        else:
            img_result = encode_image(image_path)
        image_base64 = img_result['base64']
        width = img_result['width']
        height = img_result['height']
        
        image_data.append({
            "index": i,
            "base64": image_base64,
            "width": width,
            "height": height,
            "placeholder": f"<IMAGE_PLACE_HOLDER_{i}>"
        })
        
        image_information += f"width of image_clue_{i}: {width}, height of image_clue_{i}: {height}\n"
    
    # Format question
    formatted_question = prompt_prefix.format(query=question, image_information=image_information)
    
    # Check if placeholder exists
    has_placeholders = any(f"<IMAGE_PLACE_HOLDER_{i}>" in formatted_question for i in range(len(image_path_list)))
    
    if has_placeholders:
        # Insert images at placeholder positions
        if "no_tool" in prompt_type:
            content = []
            remaining_text = formatted_question
            
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    parts = remaining_text.split(placeholder, 1)
                    
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    
                    remaining_text = parts[1]
            
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            content = []
            remaining_text = formatted_question
            
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    parts = remaining_text.split(placeholder, 1)
                    
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    i = img_data["index"]
                    content.append({"type": "text", "text": f"<image_clue_{i}>"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
                    
                    remaining_text = parts[1]
            
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
    else:
        # Handle as usual if no placeholder
        if "no_tool" in prompt_type:
            content = []
            
            for i, img_data in enumerate(image_data):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
            
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            content = []
            
            for i, img_data in enumerate(image_data):
                content.append({"type": "text", "text": f"<image_clue_{i}>"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
            
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages
 
 
def update_messages_with_execute_content(image_nums_in_input, messages, images_result, text_result, error_result, image_clue_idx):
    if error_result is None:
        new_messages = []
        image_content = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)
 
        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>\nText Result:\n{text_result}\nImage Result:\n"}]
        if images_result is not None:
            print(f"#### image_clue_index: {image_clue_idx},Image_nums_in_input: {image_nums_in_input}, len of images_result: {len(images_result)}")
            # for image_base64_item in images_result[image_clue_idx-image_nums_in_input:]:
            for image_base64_item in images_result:
                interpreter_message_images = [{"type": "text", "text": f"<image_clue_{image_clue_idx}>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_item}"}}] + [{"type": "text", "text": f"</image_clue_{image_clue_idx}>"}]
                image_content += interpreter_message_images
                image_clue_idx += 1
        else:
            image_content = [{"type": "text", "text": "None"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]
 
        interpreter_message_item = interpreter_message_text_prefix + image_content + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})
    else:
        new_messages = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)
    
        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>{error_result}"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]
    
        interpreter_message_item = interpreter_message_text_prefix + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})
 
    return new_messages, image_clue_idx
 
def update_messages_with_code(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}</code>\n"}]
    }
 
    messages.append(message_item)
    return messages
 
def update_messages_with_text(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}"}]
    }
 
    messages.append(message_item)
    return messages
 
def call_chatgpt_api(args, messages, client, max_tokens=10000, stop=None, temperature=0.6):
    """Call ChatGPT API with the given messages"""
    try:
        client_type = args.client_type
        api_name = args.api_name
    except:
        client_type = args['client_type']
        api_name = args['api_name']
 
    if client_type == "openai" or client_type == "azure":
        response = client.chat.completions.create(
            model=api_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="low",
            timeout=300
        )
        response_text = response.choices[0].message.content
        pred_stop_reason = response.choices[0].finish_reason
 
    elif client_type == "bedrock":
        def _convert_to_anthropic_messages(msgs):
            converted = []
            for msg in msgs:
                role = msg["role"]
                content = msg["content"]
                if isinstance(content, str):
                    converted.append({"role": role, "content": content})
                    continue
                blocks = []
                for block in content:
                    if block["type"] == "text":
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image_url":
                        url = block["image_url"]["url"]
                        # Parse "data:image/jpeg;base64,<data>" format
                        header, b64_data = url.split(",", 1)
                        raw = base64.b64decode(b64_data[:16])  # only need the first few bytes
                        if raw[:8] == b'\x89PNG\r\n\x1a\n':
                            media_type = "image/png"
                        elif raw[:2] == b'\xff\xd8':
                            media_type = "image/jpeg"
                        else:
                            media_type = "image/png"  # safe default since encode funcs save as PNG
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        })
                converted.append({"role": role, "content": blocks})
            return converted
 
        bedrock_messages = _convert_to_anthropic_messages(messages)
        message = client.messages.create(
            model=api_name,
            max_tokens=max_tokens,
            messages=bedrock_messages,
            temperature=1,  # required when extended thinking is enabled
            stop_sequences=stop,
            thinking={"type": "enabled", "budget_tokens": 1024},  # low reasoning effort
        )
        # When thinking is enabled, content[0] is the thinking block and content[1] is the text block.
        # Safely find the text block regardless of position.
        if isinstance(message.content, list):
            response_text = next(
                (block.text for block in message.content if getattr(block, "type", None) == "text"),
                ""
            )
        else:
            response_text = message.content
        print("CLAUDE RESPONSE: ", response_text)
 
    elif client_type == "together":
        # Strip litellm-style "together_ai/" prefix if present
        together_model = api_name.removeprefix("together_ai/")
        response = client.chat.completions.create(
            model=together_model,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort="low",
            temperature=temperature, 
            stop=stop
        )
        response_text = response.choices[0].message.content
 
    else:
        print("Your args.client_type must be one of openai, azure, bedrock, qwen, vllm, and together.")
        return None, None
 
    # Check if stop sequence is encountered
    stop_reason = None
    if stop and any(s in response_text for s in stop):
        for s in stop:
            if s in response_text:
                stop_reason = s
                break
    else:
        if client_type in ["openai", "azure", "vllm", "together"]:
            stop_reason = response.choices[0].finish_reason
        else:
            stop_reason = "stop"
    if "<code>" in response_text:
        stop_reason = "</code>"
 
    print("\n" + "="*60)
    print(f"LLM RESPONSE (stop_reason: {stop_reason}):")
    print("="*60)
    print(response_text)
    print("="*60 + "\n")
    return response_text, stop_reason
 
def evaluate_single_data(args, data, client, executor, output_dir):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        temperature = args.temperature
        api_name = args.api_name
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        temperature = args['temperature']
        api_name = args['api_name']
 
    image_path_list = data['image_path_list']
 
    if "no_tool" in prompt:
        if len(image_path_list) == 1:
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    else:
        if len(image_path_list) == 1:
            prompt = "vistool_with_img_info_v2"
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            prompt = "vistool_with_img_info_multi_image"
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    
    # Generate initial response
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None,
        temperature=temperature
    )
    
    # Handle response
    final_response = response_text
    code_execution_count = 0
    collected_functions = []
    image_clue_idx = len(image_path_list)
    
 
    max_code_steps = args.get('max_code_steps', 10) if isinstance(args, dict) else getattr(args, 'max_code_steps', 10)
 
    while True:
        # KEY THING ADDED   
        if exe_code and pred_stop_reason == "</code>" and code_execution_count < max_code_steps:
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            # Storing generated functions
            collected_functions.append(code_to_execute) 
            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                        # Save any generated images
                        if images_result:
                            try:
                                out_dir = args.output_dir if hasattr(args, 'output_dir') else args.get('output_dir', './output')
                                os.makedirs(out_dir, exist_ok=True)
                                for img_idx, img_b64 in enumerate(images_result):
                                    img_data = base64.b64decode(img_b64)
                                    img_path = os.path.join(out_dir, f"code_output_{code_execution_count}_{img_idx}.png")
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    print(f"Saved image: {img_path}")
                            except Exception as e:
                                print(f"Failed to save image: {e}")
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None
 
            messages, new_image_clue_idx = update_messages_with_execute_content(len(image_path_list), messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # Generate next response part
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None,
                temperature=temperature
            )
 
        else:
            # If we hit max_code_steps while the model was still generating code,
            # the last response_text is a partial code block — not a final answer.
            # Inject a hard-stop reminder and make one final no-stop API call.
            if exe_code and pred_stop_reason == "</code>" and code_execution_count >= max_code_steps:
                messages = update_messages_with_code(messages, response_text)
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                        "You have reached the maximum number of code execution steps. "
                        "Do NOT write any more <code> blocks. "
                        "You MUST now produce your final answer immediately using the "
                        "<final_pymc_model> block as instructed."
                    }]
                })
                response_text, _ = call_chatgpt_api(
                    args,
                    messages,
                    client,
                    max_tokens=max_tokens,
                    stop=None,  # no stop sequences — must let it finish
                    temperature=temperature
                )
 
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            tools_path = os.path.join(output_dir, f"tools_{args['idx']}.json")
            with open(tools_path, 'w') as f:
                json.dump(collected_functions, f, indent=2)
            break
       
    return messages, final_response
 
 
def evaluate_single_data_multi_images(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
 
    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # Generate initial response
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # Handle response
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # Check if code execution is needed
        if exe_code and pred_stop_reason == "</code>":
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None
 
            messages, new_image_clue_idx = update_messages_with_execute_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # Generate next response part
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )
 
        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response
 
def evaluate_single_data_video(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
 
    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # Generate initial response
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # Handle response
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # Check if code execution is needed
        if exe_code and pred_stop_reason == "</code>":
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None
 
            messages, new_image_clue_idx = update_messages_with_execute_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # Generate next response part
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )
 
        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response
 
 
# New wrapper functions for safe execution with cleanup
def evaluate_batch_with_cleanup(args, data_list, client, output_dir):
    """Wrapper function to ensure proper cleanup of resources when processing multiple items"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data(args, data, client, executor, output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing data item: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor
 
def evaluate_single_with_cleanup(args, data, client, output_dir):
    """Wrapper function for evaluating a single item with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
 
    try:
        result = evaluate_single_data(args, data, client, executor, output_dir)
        return result
    finally:
        # Ensure cleanup of persistent worker
        del executor
 
def evaluate_multi_images_with_cleanup(args, data_list, client, output_dir):
    """Wrapper function for multi-image evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_multi_images(args, data, client, executor, output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing multi-image data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor
 
def evaluate_video_with_cleanup(args, data_list, client, output_dir):
    """Wrapper function for video evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_video(args, data, client, executor, output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing video data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor