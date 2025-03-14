import copy
import json
import logging
import math
import os
import re
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Union
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin
from omegaconf import ListConfig
import logging
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


logger = logging.getLogger(__file__)

llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"



def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

class RLHFDataset(Dataset):

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,
        return_raw_chat=False,
        truncation='error',
        filter_overlong_prompts=False,
        #transform=None,
        slice_config=None,
        llm_type="qwen",
        patch_size=14,
        query_nums=64,
        batch_vision=True,
    ):
        super(RLHFDataset, self).__init__()
        
        if not isinstance(parquet_files, (List, ListConfig)) and parquet_files is not None:
            parquet_files = [parquet_files]
            
        self.parquet_files = copy.deepcopy(parquet_files) if parquet_files is not None else None
        self.original_parquet_files = copy.deepcopy(parquet_files) if parquet_files is not None else None
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts
        
        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        
        self.transform = build_transform()
        self.slice_config = {"max_slice_nums": 9, "patch_size": patch_size, "scale_resolution": 448} 
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
            
    def _download(self, use_origin_parquet=False):
        """与RLHFDataset保持一致的下载方法"""
        pass
        
    def _read_files_and_tokenize(self):
        import pandas as pd
        
        dataframes = []
        for parquet_file in self.parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error reading parquet file {parquet_file}: {e}")
                
        if dataframes:
            self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f'dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            item: dict = self.dataframe.iloc[i].to_dict()
            images_dict = {}
            if len(item[self.image_key]) == 1:
                img_bytes = item.pop(self.image_key)[0]["bytes"]
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                images_dict = { "<image>" : image}
            else:
                raise NotImplementedError
                #TODO handle multi images
            #print("images_dict:", images_dict) 

            ret = preprocess(
                images_dict,
                item[self.prompt_key],
                self.tokenizer,
                self.transform,
                query_nums=self.query_nums,
                slice_config=self.slice_config,
                llm_type=self.llm_type,
                patch_size=self.patch_size,
                batch_vision=self.batch_vision,
                max_length=self.max_prompt_length,
                truncation=self.truncation
            )
            
            prompt_with_chat_template = self.tokenizer.apply_chat_template(item[self.prompt_key], add_generation_prompt=True, tokenize=False)
            raw_prompt = prompt_with_chat_template.replace('<image>', '(<image>./</image>)') # minicpm-o vllm placeholder

            row_dict = {}
            
            row_dict["input_ids"] = ret["input_ids"]
            row_dict["attention_mask"] = ret["attention_mask"]
            row_dict["position_ids"] = ret["position_ids"]
            row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False) # for vllm
            row_dict["reward_model"] = item["reward_model"]
            row_dict["data_source"] = item["data_source"]


            row_dict["multi_modal_inputs"] = {}
            
            if "pixel_values" in ret and len(ret["pixel_values"]) > 0:
                row_dict["multi_modal_inputs"]["pixel_values"] = ret["pixel_values"]
                
                if "tgt_sizes" in ret and len(ret["tgt_sizes"]) > 0:
                    row_dict["multi_modal_inputs"]["tgt_sizes"] = ret["tgt_sizes"]
                
                if "image_bound" in ret and len(ret["image_bound"]) > 0:
                    row_dict["multi_modal_inputs"]["image_bound"] = ret["image_bound"]
                
                # 添加原始图像数据
                row_dict["multi_modal_data"] = {"image": [img for img in images_dict.values()]} # for vllm
            
            
            # 添加索引信息
            row_dict["index"] = i
            
            
        except Exception as e:
            logger.error(f"data getitem error: {str(e)}")
            return self.__getitem__(random.randint(0, len(self)))
            
        return row_dict

        
def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def build_image_bound(input_ids, tokenizer, new_schema=True):
    if new_schema:
        start_cond = (input_ids == tokenizer.im_start_id) | (input_ids == tokenizer.slice_start_id)
        end_cond = (input_ids == tokenizer.im_end_id) | (input_ids == tokenizer.slice_end_id)
    else:
        start_cond = (input_ids == tokenizer.im_start_id)
        end_cond = (input_ids == tokenizer.im_end_id)
    image_start_tokens = torch.where(start_cond)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(end_cond)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")
    if len(image_start_tokens) > 0:
        image_bound = torch.hstack(
            [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
        )
    else:
        image_bound = []
    return image_bound


def preprocess(
    images_dict,
    conversations,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
    max_length=2048,
    truncation='error',
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    new_schema = False
    use_image_id = False
    if llm_type=='qwen':
        new_schema = True
        use_image_id = True
    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0 
    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder
            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])
                if use_image_id:
                    image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                    image_id_cnt += 1
                image_placeholder += get_grid_placeholder(
                    tokenizer, best_grid, query_nums, new_schema = new_schema)
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder
    
    images = [transform(i) for i in images]
    
    if len(images_dict) == 1 and "<image>" in images_dict:       
        if "<image>" in conversations[0]["content"]:
            conversations[0]["content"] = conversations[0]["content"].replace(
                "<image>", image_placeholder
            )
        else:
            conversations[0]["content"] = (
                image_placeholder + "\n" + conversations[0]["content"]
            )
    else:
        pattern = r'<image_\d+>'
        new_conversations = []
        for conversation in conversations:
            content = conversation['content']
            parts = re.split(f'({pattern})', content)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                if re.match(pattern, part):  
                    if part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part] 
                    else:
                        raise Exception(f"not found {part} in image dict")
            conversation['content'] = '\n'.join(parts)
            new_conversations.append(conversation)
        conversations = new_conversations

    #TODO change role in conversation for different llm
    prompt_with_chat_template = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                        tokenizer=tokenizer,
                                                                        max_length=max_length,
                                                                        pad_token_id=tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation=truncation)
    position_ids = compute_position_id_with_mask(attention_mask)
    image_bound = build_image_bound(input_ids[0], tokenizer, new_schema)

    input_dict = {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
        "image_bound": image_bound,
    }

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
        (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = (
            tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
        )
    else:
        image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = '\n'.join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches
