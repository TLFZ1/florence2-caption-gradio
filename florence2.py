# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# 导入所需的库
# ----------------------------------------------------------------------
import os  # 用于与操作系统交互，如文件路径操作
import torch  # PyTorch深度学习框架
import time  # 时间相关的功能
import importlib.metadata  # 用于检查已安装的包版本
import re  # 正则表达式库，用于文本清理
from pathlib import Path  # 面向对象的文件系统路径库，方便地处理文件和目录
from PIL import Image  # Python Imaging Library (PIL) 的一个分支，用于打开、操作和保存多种图像文件格式
from transformers import AutoProcessor, AutoModelForCausalLM  # Hugging Face的transformers库，用于加载模型和处理器
from torch.utils.data import Dataset, DataLoader  # PyTorch的数据加载工具
from tqdm import tqdm  # 用于显示美观的进度条
import gc  # 垃圾回收模块，用于手动管理内存
import gradio as gr # 导入Gradio库

# ======================================================================
# --- 1. Gradio界面默认值配置区 ---
# ======================================================================
# 在这里修改变量，Gradio界面的默认值会自动更新

# --- 路径配置 ---
DEFAULT_IMAGE_FOLDER = r""  # 默认的输入图片文件夹路径
DEFAULT_OUTPUT_FOLDER = r"" # 默认的输出文件夹路径
DEFAULT_MODEL_STORAGE_DIR = "./florence2_models"  # 模型下载和缓存的本地目录

# --- 模型配置 ---
# 可用的模型列表，用于Gradio下拉菜单
MODEL_ID_LIST = [
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-base-ft",
    "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
]
DEFAULT_MODEL_ID = "microsoft/Florence-2-large"  # 默认使用的Hugging Face模型ID

# --- 功能开关 ---
DEFAULT_CLEANUP_EXISTING = True  # 是否在开始时清理已存在的.txt文件
DEFAULT_SKIP_EXISTING = True  # 是否跳过已存在的.txt文件，实现断点续传

# --- 性能与精度配置 ---
DEFAULT_BATCH_SIZE = 15  # 批处理大小
DEFAULT_NUM_WORKERS = 6  # 数据加载进程数
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"  # 注意力机制实现
DEFAULT_PRECISION = "fp32"  # 计算精度

# --- 生成效果配置 ---
DEFAULT_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"  # 默认的任务提示词
DEFAULT_MAX_NEW_TOKENS = 1024  # 生成文本的最大长度
DEFAULT_NUM_BEAMS = 3  # 集束搜索的宽度
DEFAULT_EARLY_STOPPING = False  # 是否启用提前停止策略

# --- 全局变量，用于缓存模型和处理器，避免重复加载 ---
loaded_models_cache = {}

# ======================================================================
# --- 2. 核心功能代码 ---
# ======================================================================

def print_acceleration_status(device_str, attn_implementation):
    """检查并打印当前PyTorch环境的SDPA/FlashAttention加速状态。"""
    console_output = "\n--- 加速状态诊断 ---\n"
    device = torch.device(device_str)
    if device.type != 'cuda':
        console_output += f"🐌 设备: {device.type.upper()} / 加速不可用\n"
        console_output += "---------------------\n"
        return console_output
    
    console_output += f"设备: {torch.cuda.get_device_name(0)} (CUDA)\n"
    
    try:
        importlib.metadata.version('flash-attn')
        flash_attn_installed = True
    except importlib.metadata.PackageNotFoundError:
        flash_attn_installed = False

    if attn_implementation == 'flash_attention_2':
        if flash_attn_installed:
            console_output += "✅ 加速方案: 强制使用 FlashAttention 2\n"
        else:
            console_output += "❌ 加速方案: 强制使用 FlashAttention 2 (但配置可能失败)\n"
            console_output += "   警告: 您指定使用 flash_attention_2，但未检测到 'flash-attn' 包！\n"
        console_output += "---------------------\n"
        return console_output

    console_output += f"✅ 加速方案: SDPA (Pytorch原生实现)\n"
    if flash_attn_installed and torch.backends.cuda.flash_sdp_enabled():
        console_output += "   后端内核: ⚡ FlashAttention (已安装独立的 flash-attn 包)\n"
    elif torch.backends.cuda.flash_sdp_enabled():
        console_output += "   后端内核: 🧠 PyTorch 内置 FlashAttention\n"
    elif torch.backends.cuda.mem_efficient_sdp_enabled():
        console_output += "   后端内核: 🧠 PyTorch 内置内存优化方案\n"
    else:
        console_output += "   后端内核: 🧮 PyTorch 原生数学实现\n"
    console_output += "---------------------\n"
    return console_output

def cleanup_caption(text: str, task_prompt: str) -> str:
    """清理文本，移除特殊词，并将所有内容合并成一个干净的单行字符串。"""
    if not isinstance(text, str):
        return ""
    # 定义需要移除的特殊词列表
    tokens_to_remove = ["<pad>", "</s>", "<s>", task_prompt]
    # 循环替换
    for token in tokens_to_remove:
        text = text.replace(token, "")
    # 将换行符替换为空格
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    # 将多个连续空格合并为一个
    text = re.sub(r'\s+', ' ', text)
    # 返回清理并去除首尾空格后的文本
    return text.strip()

def cleanup_existing_files(output_folder, task_prompt, progress=gr.Progress(track_tqdm=True)):
    """遍历输出文件夹，根据cleanup_caption的规则清理所有.txt文件。"""
    output_path = Path(output_folder)
    # 检查输出文件夹是否存在
    if not output_path.is_dir():
        return "状态：输出文件夹不存在，跳过清理。", ""
    # 查找所有txt文件
    txt_files = list(output_path.glob("*.txt"))
    if not txt_files:
        return "状态：未找到任何.txt文件，无需清理。", ""
    
    cleaned_count = 0
    console_log = ""
    # 使用tqdm显示清理进度
    for file_path in progress.tqdm(txt_files, desc="清理旧文件"):
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 清理内容
            cleaned_content = cleanup_caption(content, task_prompt)
            # 如果内容有变化，则写回文件
            if content != cleaned_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                cleaned_count += 1
                console_log += f"已清理: {file_path.name}\n"
        except Exception as e:
            console_log += f"处理文件 {file_path} 时出错: {e}\n"
    return f"状态：旧文件清理完成！共 {cleaned_count} 个文件被更新。", console_log

class ImageDataset(Dataset):
    """自定义的PyTorch数据集类，用于加载图片。"""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根据索引获取单个样本
        path = self.image_paths[idx]
        try:
            # 打开图片并转换为RGB格式
            image = Image.open(path).convert("RGB")
            # 返回图片对象和其路径
            return image, str(path)
        except Exception as e:
            # 如果加载失败，则打印错误并返回None
            print(f"警告：加载图片时出错 {path}, 错误: {e}")
            return None

def collate_fn_skip_none(batch):
    """自定义的collate_fn，用于在DataLoader组合批次时，过滤掉加载失败的None项。"""
    return [item for item in batch if item is not None]

def find_image_files(folder_path, recursive):
    """使用pathlib查找指定目录下的所有图片文件。"""
    dir_path = Path(folder_path)
    if not dir_path.is_dir(): return []
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif']
    glob_pattern = '**/*' if recursive else '*'
    return [p for p in dir_path.glob(glob_pattern) if p.is_file() and p.suffix.lower() in extensions]

# ======================================================================
# --- 3. Gradio 核心处理函数 ---
# ======================================================================

def run_batch_tagging_ui(
    # 从Gradio界面接收所有参数
    image_folder, output_folder, model_storage_dir,
    model_id, precision, attn_implementation,
    batch_size, num_workers,
    task_prompt, max_new_tokens, num_beams, early_stopping,
    cleanup_existing, skip_existing,
    progress=gr.Progress(track_tqdm=True) # Gradio进度条对象
):
    """
    Gradio界面点击“开始”后执行的核心函数。这是一个生成器函数，会逐步yield状态更新。
    """
    # 初始化全局模型缓存和UI输出
    global loaded_models_cache
    status_message = "状态：任务准备中..."
    console_output = ""
    yield status_message, console_output

    # 参数校验
    if not image_folder or not os.path.isdir(image_folder):
        raise gr.Error("错误：输入文件夹路径无效或不存在。")
    if not output_folder:
        raise gr.Error("错误：必须指定输出文件夹路径。")
    
    # 创建输出文件夹并执行清理（如果需要）
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    if cleanup_existing:
        status_message, cleanup_log = cleanup_existing_files(output_folder, task_prompt, progress)
        console_output += cleanup_log
        yield status_message, console_output

    # --- 智能模型加载/卸载逻辑 ---
    status_message = "状态：正在准备模型..."
    yield status_message, console_output
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console_output += print_acceleration_status(device, attn_implementation)
    yield status_message, console_output

    # 使用模型配置创建唯一的键，用于缓存
    new_model_key = f"{model_id}-{precision}-{attn_implementation}"
    current_model_key = next(iter(loaded_models_cache)) if loaded_models_cache else None

    # 如果请求的模型与当前加载的模型不同，或者没有任何模型被加载
    if new_model_key != current_model_key:
        # 1. 卸载旧模型 (如果存在)
        if current_model_key is not None:
            console_output += f"\n正在从显存卸载旧模型: {current_model_key}...\n"
            yield status_message, console_output
            old_model, old_processor = loaded_models_cache.pop(current_model_key)
            del old_model
            del old_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            console_output += "旧模型已卸载。\n"
            yield status_message, console_output
        
        # 2. 加载新模型
        try:
            console_output += f"首次加载新模型 {model_id}...\n"
            yield status_message, console_output
            torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
            if precision == "bf16" and device == 'cuda' and not torch.cuda.is_bf16_supported():
                console_output += "警告：GPU不支持bf16，自动切换到fp16。\n"
                torch_dtype = torch.float16
            
            # 从Hugging Face Hub加载模型和处理器
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=model_storage_dir, attn_implementation=attn_implementation, 
                torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device).eval() # .eval()切换到推理模式
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_storage_dir, trust_remote_code=True)
            
            # 3. 将新模型存入缓存
            loaded_models_cache[new_model_key] = (model, processor)
            console_output += "新模型加载成功！\n"
            yield status_message, console_output
        except Exception as e:
            raise gr.Error(f"模型加载失败: {e}")
    else:
        # 如果请求的模型已加载，直接使用
        console_output += "使用已缓存的模型。\n"
        yield status_message, console_output

    # 从缓存中获取当前要使用的模型
    model, processor = loaded_models_cache[new_model_key]

    # --- 扫描并筛选要处理的图片文件 ---
    status_message = "状态：正在扫描并筛选图片..."
    yield status_message, console_output
    
    all_image_paths = find_image_files(image_folder, recursive=True)
    if not all_image_paths:
        yield "完成：在输入文件夹中未找到任何图片。", console_output
        return
        
    if skip_existing:
        # 如果启用“跳过”，则只保留那些没有对应.txt文件的图片
        unprocessed_images = [p for p in all_image_paths if not (output_folder_path / (p.stem + ".txt")).exists()]
    else:
        unprocessed_images = all_image_paths
            
    total_images = len(all_image_paths)
    processed_count = total_images - len(unprocessed_images)
    
    console_output += f"总共找到 {total_images} 张图片。\n"
    if skip_existing:
        console_output += f"其中 {processed_count} 张已有标签，将被跳过。\n"
    
    if not unprocessed_images:
        yield "完成：所有图片均已处理完毕！", console_output
        return

    console_output += f"即将处理 {len(unprocessed_images)} 张新图片...\n"
    yield status_message, console_output

    # 创建数据集和数据加载器
    dataset = ImageDataset(unprocessed_images)
    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        collate_fn=collate_fn_skip_none, pin_memory=True if device == "cuda" else False
    )

    # --- 批量处理循环 ---
    start_time = time.time()
    newly_processed_count = 0
    for batch in progress.tqdm(dataloader, desc="批量处理中"):
        if not batch: continue # 如果批次为空，跳过
        images, batch_image_paths = zip(*batch)
        try:
            # 在no_grad上下文中执行所有PyTorch操作，以防止显存泄漏
            with torch.no_grad():
                task_prompts = [task_prompt] * len(images)
                inputs = processor(text=task_prompts, images=images, return_tensors="pt").to(device)
                
                # 模型生成文本
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=int(max_new_tokens), num_beams=int(num_beams), early_stopping=early_stopping
                )
                # 解码生成的ID为文本
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            # 遍历批次中的每个结果
            for idx, single_text in enumerate(generated_texts):
                current_image = images[idx]
                current_path = Path(batch_image_paths[idx])

                # 智能判断是否需要为后处理器添加image_size参数
                post_process_args = {'task': task_prompt}
                if "MiaoshouAI" in model_id or "-ft" in model_id:
                    post_process_args['image_size'] = current_image.size

                # 后处理生成结果
                parsed_answer = processor.post_process_generation(single_text, **post_process_args)
                raw_caption = parsed_answer.get(task_prompt, "")
                # 清理最终的描述文本
                caption = cleanup_caption(raw_caption, task_prompt)
                
                # 保存到文件
                if caption:
                    output_filename = current_path.stem + ".txt"
                    output_filepath = output_folder_path / output_filename
                    with open(output_filepath, "w", encoding="utf-8") as f:
                        f.write(caption)
                    console_output += f"已处理: {current_path.name}\n"
                    newly_processed_count += 1
        except Exception as e:
            # 捕获并记录错误
            console_output += f"\n错误：处理批次时发生意外: {e}\n"
            import traceback
            console_output += traceback.format_exc() + "\n"
        finally:
            # 手动清理循环中的变量和CUDA缓存，防止显存泄漏
            del images, batch_image_paths, task_prompts, inputs, generated_ids, generated_texts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 流式更新UI状态
        yield f"状态：处理中...({newly_processed_count}/{len(unprocessed_images)})", console_output

    # 任务结束，计算并显示统计信息
    end_time = time.time()
    total_time = end_time - start_time
    images_per_second = newly_processed_count / total_time if total_time > 0 else 0
    
    final_summary = (f"完成！\n本次运行处理了 {newly_processed_count} 张新图片。\n"
        f"总耗时: {total_time:.2f} 秒\n"
        f"平均速度: {images_per_second:.2f} 张/秒")
    yield final_summary, console_output

# ======================================================================
# --- 4. Gradio 界面定义 ---
# ======================================================================

def create_ui():
    """创建并定义Gradio界面的所有组件和布局。"""
    # 使用Gradio的默认主题，根据用户反馈，该主题在其系统上显示为橙色
    with gr.Blocks(title="Florence-2 批量打标工具", theme=gr.themes.Default()) as demo:
        # 顶层标题
        gr.Markdown("# 🖼️ Florence-2 批量图片描述生成工具")
        gr.Markdown("一个基于Florence-2系列模型的高性能、可视化的批量图片打标工具。")

        # 主布局，分为左右两列
        with gr.Row():
            # 左侧主设置区
            with gr.Column(scale=2):
                # 使用可折叠的Accordion组件来组织UI，使其更整洁
                with gr.Accordion("主要设置", open=True):
                    gr.Markdown("### **1. 路径设置**")
                    with gr.Row():
                        image_folder_input = gr.Textbox(label="输入图片文件夹", value=DEFAULT_IMAGE_FOLDER, placeholder="例如: D:\\dataset\\images")
                        output_folder_input = gr.Textbox(label="输出标签文件夹", value=DEFAULT_OUTPUT_FOLDER, placeholder="例如: D:\\dataset\\captions")
                    
                    gr.Markdown("### **2. 模型与精度**")
                    with gr.Row():
                        model_id_input = gr.Dropdown(MODEL_ID_LIST, value=DEFAULT_MODEL_ID, label="选择模型")
                        precision_input = gr.Dropdown(["fp32", "fp16", "bf16"], value=DEFAULT_PRECISION, label="计算精度")
                    
                    attn_implementation_input = gr.Dropdown(["sdpa", "flash_attention_2", "eager"], value=DEFAULT_ATTN_IMPLEMENTATION, label="注意力机制实现", info="推荐使用 'sdpa'。")
                    model_storage_dir_input = gr.Textbox(label="模型缓存目录", value=DEFAULT_MODEL_STORAGE_DIR)

                with gr.Accordion("生成效果设置", open=True):
                    task_prompt_input = gr.Textbox(label="任务提示词 (Task Prompt)", value=DEFAULT_TASK_PROMPT)
                    with gr.Row():
                        max_new_tokens_input = gr.Slider(64, 2048, value=DEFAULT_MAX_NEW_TOKENS, step=64, label="最大生成长度")
                        num_beams_input = gr.Slider(1, 10, value=DEFAULT_NUM_BEAMS, step=1, label="集束搜索宽度 (Num Beams)")
                    early_stopping_input = gr.Checkbox(label="启用提前停止 (Early Stopping)", value=DEFAULT_EARLY_STOPPING, info="追求更简洁、高效的输出时勾选。")
                
                with gr.Accordion("运行与功能设置", open=False):
                    with gr.Row():
                        batch_size_input = gr.Slider(1, 64, value=DEFAULT_BATCH_SIZE, step=1, label="批处理大小 (Batch Size)")
                        num_workers_input = gr.Slider(0, 16, value=DEFAULT_NUM_WORKERS, step=1, label="数据加载进程数", info="重要：为使“强制停止”按钮有效，此值必须为 0。")
                    with gr.Row():
                        cleanup_existing_input = gr.Checkbox(label="开始前清理旧标签文件", value=DEFAULT_CLEANUP_EXISTING)
                        skip_existing_input = gr.Checkbox(label="跳过已存在的标签文件", value=DEFAULT_SKIP_EXISTING)
                
                # 控制按钮
                with gr.Row():
                    start_button = gr.Button("开始批量打标", variant="primary", size="lg")
                    stop_button = gr.Button("强制停止", variant="stop", size="lg", visible=False) # 默认不可见

            # 右侧状态和日志区
            with gr.Column(scale=1):
                gr.Markdown("### **运行状态与日志**")
                status_output = gr.Textbox(label="当前状态", interactive=False, lines=3, max_lines=3)
                console_output = gr.Textbox(label="处理日志", interactive=False, lines=25, max_lines=25, autoscroll=True)

        # --- 事件处理逻辑 ---
        # 将所有输入UI组件收集到一个列表中，方便传递给核心处理函数
        run_inputs = [
            image_folder_input, output_folder_input, model_storage_dir_input,
            model_id_input, precision_input, attn_implementation_input,
            batch_size_input, num_workers_input,
            task_prompt_input, max_new_tokens_input, num_beams_input, early_stopping_input,
            cleanup_existing_input, skip_existing_input
        ]
        
        # 定义点击“开始”按钮后的行为
        def start_running():
            return {start_button: gr.update(visible=False), stop_button: gr.update(visible=True)}
        
        # 定义任务结束或手动停止后的行为
        def stop_running(status_message=""):
            update_dict = {start_button: gr.update(visible=True), stop_button: gr.update(visible=False)}
            if status_message:
                update_dict[status_output] = gr.update(value=status_message)
            return update_dict

        # “开始”按钮的点击事件链
        process_event = start_button.click(
            fn=start_running, outputs=[start_button, stop_button] # 第一步：切换按钮可见性
        ).then(
            fn=run_batch_tagging_ui, inputs=run_inputs, outputs=[status_output, console_output] # 第二步：调用核心处理函数
        ).then(
            fn=lambda: stop_running("任务完成！"), outputs=[start_button, stop_button, status_output] # 第三步：任务结束后恢复按钮
        )
        
        # “停止”按钮的点击事件
        stop_button.click(
            fn=lambda: stop_running("状态：任务已手动停止。"), 
            outputs=[start_button, stop_button, status_output], 
            cancels=[process_event] # 关键：取消正在运行的process_event事件链
        )
        
        return demo

# ======================================================================
# --- 5. 主程序入口 ---
# ======================================================================

# 【核心修改】将UI创建和启动逻辑包裹在 if __name__ == "__main__": 中
# 这是解决Windows下多进程错误的标准做法
if __name__ == "__main__":
    app = create_ui()
    app.launch(inbrowser=True) # inbrowser=True 会自动在浏览器中打开新标签页